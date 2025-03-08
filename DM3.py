from my_data_structs import *
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import tqdm
import time

# -------------------------------
# Risk Correlation and Distance Computations
# -------------------------------
def compute_risk_corr_np(risk1, risk2, alpha=1e-4):
    """
    Computes a modified correlation between two risk series that captures jump sizes.
    """
    diff1 = np.diff(risk1)
    diff2 = np.diff(risk2)
    similarity = np.exp(-alpha * np.abs(diff1 - diff2))  # values in (0,1]
    sign = np.where(diff1 * diff2 >= 0, 1, -1)
    contributions = sign * similarity
    return np.sum(contributions)

def compute_risk_corr_matrix(instance, alpha=1e-4):
    """
    Computes a normalized correlation matrix from risk series in instance.
    Normalized correlation: norm_corr(i,j) = corr(i,j) / sqrt(corr(i,i)*corr(j,j))
    """
    interventions = instance.interventions
    keys = list(interventions.keys())
    n = len(keys)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        risk_i = np.array(interventions[keys[i]].overall_mean_risk)
        for j in range(i, n):
            risk_j = np.array(interventions[keys[j]].overall_mean_risk)
            corr = compute_risk_corr_np(risk_i, risk_j, alpha)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    norm_corr_matrix = np.zeros_like(corr_matrix)
    for i in range(n):
        for j in range(n):
            norm_corr_matrix[i, j] = corr_matrix[i, j] / np.sqrt(corr_matrix[i, i] * corr_matrix[j, j])
    return keys, norm_corr_matrix

def transform_corr_to_distance(corr_matrix, method="linear"):
    """
    Transforms a correlation matrix with entries in [-1, 1] into a distance matrix in [0, 1].
    High correlation (r=1) gives d=0 and low correlation (r=-1) gives d=1.

    Parameters:
        corr_matrix (array-like): Matrix with correlations in [-1, 1].
        method (str): Transformation method. Options include:
            - "linear":     d = (1 - r) / 2
            - "sqrt":       d = sqrt((1 - r) / 2)
            - "arccos":     d = arccos(r) / π
            - "logistic":   d = (f(r) - f(1)) / (f(-1) - f(1))
                            where f(r)=1/(1+exp(a*r)) with a default steepness a=2.
            - "exponential":d = (exp(-k*r) - exp(-k)) / (exp(k) - exp(-k)) with k=1.
            - "power2":     d = ((1 - r)/2)^2
            - "power1/3":   d = ((1 - r)/2)^(1/3)
            - "arctan":     d = arctan(alpha*(1 - r)) / arctan(2*alpha) with alpha=1.
            - "sine":       d = sin((π/4)*(1 - r))
            
    Returns:
        numpy.ndarray: A distance matrix with values in [0, 1].
    """
    corr_matrix = np.array(corr_matrix)
    
    # Validate correlation range.
    if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
        raise ValueError("All correlation values must be in the range [-1, 1].")
    
    if method == "linear":
        distance_matrix = (1 - corr_matrix) / 2
    elif method == "sqrt":
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    elif method == "arccos":
        distance_matrix = np.arccos(corr_matrix) / np.pi
    elif method == "logistic":
        a = 2  # steepness parameter
        f = lambda r: 1 / (1 + np.exp(a * r))
        f1 = f(1)
        f_neg1 = f(-1)
        distance_matrix = (f(corr_matrix) - f1) / (f_neg1 - f1)
    elif method == "exponential":
        k = 1
        distance_matrix = (np.exp(-k * corr_matrix) - np.exp(-k)) / (np.exp(k) - np.exp(-k))
    elif method == "power2":
        distance_matrix = ((1 - corr_matrix) / 2)**2
    elif method == "power1/3":
        distance_matrix = ((1 - corr_matrix) / 2)**(1/3)
    elif method == "arctan":
        alpha = 1
        distance_matrix = np.arctan(alpha * (1 - corr_matrix)) / np.arctan(alpha * 2)
    elif method == "sine":
        distance_matrix = np.sin((np.pi/4) * (1 - corr_matrix))
    else:
        raise ValueError(f"Unknown distance transformation method: {method}")
    
    return distance_matrix

def compute_distance_matrix(instance, alpha=1e-4, method="linear"):
    """
    Computes a distance matrix from the normalized correlation matrix.
    After transforming, distances (except the diagonal) are normalized to the range [epsilon, 1].
    """
    keys, norm_corr_matrix = compute_risk_corr_matrix(instance, alpha)
    distance_matrix = transform_corr_to_distance(norm_corr_matrix, method=method)
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    d_min = np.min(distance_matrix[mask])
    d_max = np.max(distance_matrix[mask])
    epsilon = 1e-6
    # Rescale distances to fully use the range [epsilon, 1]
    distance_matrix[mask] = epsilon + (1 - epsilon) * (distance_matrix[mask] - d_min) / (d_max - d_min)
    return keys, distance_matrix

# -------------------------------
# Weighted Stress Computation
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

# -------------------------------
# Deep Neural Network for MDS
# -------------------------------
class DeepNet1(nn.Module):
    def __init__(self, n_points, dropout=0.2):
        """
        A deep neural network with 5 layers that gradually reduces
        the dimensionality from n_points to 2.
        
        Parameters:
          - n_points: The input dimension (number of points).
          - dropout: Dropout rate for regularization.
        """
        super(DeepNet1, self).__init__()
        self.fc1 = nn.Linear(n_points, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) 
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc5 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

class DeepNet2(nn.Module):
    def __init__(self, n_points, dropout=0.3):
        """
        A deep neural network with 8 layers that gradually reduces
        the dimensionality from n_points to 2.
        
        Parameters:
          - n_points: The input dimension.
          - dropout: Dropout rate for regularization.
        """
        super(DeepNet2, self).__init__()
        
        self.fc1 = nn.Linear(n_points, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        
        self.fc7 = nn.Linear(32, 16)
        self.bn7 = nn.BatchNorm1d(16)
        
        self.fc8 = nn.Linear(16, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(F.silu(self.bn1(self.fc1(x))))
        x = self.dropout(F.silu(self.bn2(self.fc2(x))))
        x = self.dropout(F.silu(self.bn3(self.fc3(x))))
        x = self.dropout(F.silu(self.bn4(self.fc4(x))))
        x = self.dropout(F.silu(self.bn5(self.fc5(x))))
        x = self.dropout(F.silu(self.bn6(self.fc6(x))))
        x = self.dropout(F.silu(self.bn7(self.fc7(x))))
        x = self.fc8(x)
        return x
    

class DeepNet3(nn.Module):
    def __init__(self, n_points, dropout=0.3):
        """
        A deep neural network with 12 layers:
          - 1 input layer mapping n_points to 1024,
          - 8 intermediate layers with residual connections (all with 1024 dimensions),
          - 4 final layers for a steeper reduction to the 2-dimensional output.
          
        Parameters:
          - n_points: The input dimension.
          - dropout: Dropout rate for regularization.
        """
        super(DeepNet3, self).__init__()
        
        # Initial layer: input to high-dimensional representation.
        self.fc_in = nn.Linear(n_points, 1024)
        self.bn_in = nn.BatchNorm1d(1024)
        
        # 8 intermediate layers (residual blocks) with constant dimensionality.
        # Using ModuleList to store a sequence of identical blocks.
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU()
            )
            for _ in range(8)
        ])
        
        # 4 layers of dimensionality reduction (steeper drop):
        # Reducing dimensions from 1024 -> 256 -> 64 -> 16 -> 2.
        self.fc9  = nn.Linear(1024, 256)
        self.bn9  = nn.BatchNorm1d(256)
        
        self.fc10 = nn.Linear(256, 64)
        self.bn10 = nn.BatchNorm1d(64)
        
        self.fc11 = nn.Linear(64, 16)
        self.bn11 = nn.BatchNorm1d(16)
        
        self.fc12 = nn.Linear(16, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input mapping
        x = self.dropout(F.silu(self.bn_in(self.fc_in(x))))
        
        # Pass through 8 residual layers.
        # Each block: output = F(x) + x.
        for layer in self.res_layers:
            residual = x
            out = layer(x)
            x = self.dropout(out + residual)
        
        # Dimensionality reduction layers.
        x = self.dropout(F.silu(self.bn9(self.fc9(x))))
        x = self.dropout(F.silu(self.bn10(self.fc10(x))))
        x = self.dropout(F.silu(self.bn11(self.fc11(x))))
        x = self.fc12(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        """
        A residual block that contains two linear layers with batch normalization,
        SiLU activation, dropout, and a skip connection.
        If in_features != out_features, a linear projection is used for the shortcut.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # If dimensions differ, project the input to the correct size.
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.silu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.silu(out)
        out = self.dropout(out)
        
        # Add the skip connection.
        out += identity
        return out

class DeepNet(nn.Module):
    def __init__(self, n_points, dropout=0.3):
        """
        A deep residual network where each original layer is replaced by a residual block.
        This model has 8 residual blocks (each with 2 linear layers), making it 16 layers deep,
        and gradually reduces the dimensionality from n_points to 2.
        
        Parameters:
          - n_points: The input dimension.
          - dropout: Dropout rate for regularization.
        """
        super(DeepNet, self).__init__()
        
        self.block1 = ResidualBlock(n_points, 1024, dropout)
        self.block2 = ResidualBlock(1024, 512, dropout)
        self.block3 = ResidualBlock(512, 256, dropout)
        self.block4 = ResidualBlock(256, 128, dropout)
        self.block5 = ResidualBlock(128, 64, dropout)
        self.block6 = ResidualBlock(64, 32, dropout)
        self.block7 = ResidualBlock(32, 16, dropout)
        self.block8 = ResidualBlock(16, 2, dropout)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNetConv(nn.Module):
    def __init__(self, n_points, dropout=0.3):
        """
        A deep neural network that:
          - Maps the input to a 1024-dimensional space.
          - Uses 4 residual blocks (fully connected) to refine features.
          - Reshapes the vector and applies 4 convolutional layers
            with stride=2 for gradual reduction.
          - Uses adaptive pooling followed by 2 fully connected layers
            to produce a 2-dimensional output.
          
        Parameters:
          - n_points: The input dimension.
          - dropout: Dropout rate for regularization.
        """
        super(DeepNetConv, self).__init__()
        
        # Initial mapping: input to 1024 dimensions.
        self.fc_in = nn.Linear(n_points, 1024)
        self.bn_in = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(dropout)
        
        # 4 residual blocks that preserve the 1024-dim feature space.
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU()
            ) for _ in range(4)
        ])
        
        # Convolutional layers for gradual reduction.
        # We first reshape the 1024-dim vector to (batch, 1, 1024)
        # so that we can use 1D convolutions.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn_conv1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_conv3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn_conv4 = nn.BatchNorm1d(256)
        
        # Adaptive pooling to force the sequence length to 1.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final two fully connected layers.
        # Here, we first reduce from 256 to an intermediate 64,
        # then map to the final 2-dimensional output.
        self.fc_final1 = nn.Linear(256, 64)
        self.fc_final2 = nn.Linear(64, 2)
        
    def forward(self, x):
        # x: (batch, n_points)
        x = self.dropout(F.silu(self.bn_in(self.fc_in(x))))
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            out = block(x)
            x = self.dropout(out + residual)
        
        # Reshape for convolution: (batch, 1024) -> (batch, 1, 1024)
        x = x.unsqueeze(1)
        
        # Convolutional blocks with gradual reduction.
        x = self.dropout(F.silu(self.bn_conv1(self.conv1(x))))
        x = self.dropout(F.silu(self.bn_conv2(self.conv2(x))))
        x = self.dropout(F.silu(self.bn_conv3(self.conv3(x))))
        x = self.dropout(F.silu(self.bn_conv4(self.conv4(x))))
        
        # Adaptive pooling to get shape (batch, 256, 1)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)  # Now (batch, 256)
        
        # Final fully connected layers.
        x = self.dropout(F.silu(self.fc_final1(x)))
        x = self.fc_final2(x)
        return x


# Using symmetry

class ResidualBlockSymm(nn.Module):
    """
    A residual block for per-point features.
    Expects input of shape (B, n_points, features) and applies a shared MLP on each point.
    """
    def __init__(self, features, dropout=0.3):
        super(ResidualBlockSymm, self).__init__()
        self.fc1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.fc2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, n_points, features)
        residual = x
        B, n_points, features = x.shape
        x = x.view(B * n_points, features)
        x = self.dropout(F.silu(self.bn1(self.fc1(x))))
        x = self.bn2(self.fc2(x))
        x = x.view(B, n_points, features)
        return F.silu(x + residual)

class DeepNetSymm(nn.Module):
    """
    Reconstructs the original points from a symmetric distance matrix.

    This architecture uses two branches:
      - Branch 1 processes each row of the matrix with shared weights and residual blocks,
        then aggregates features via 1D convolutions.
      - Branch 2 extracts the unique upper-triangular elements.
    These are fused and decoded to yield the final point coordinates.

    Parameters:
      - n_points: Number of points in the distance matrix (matrix is n_points x n_points).
      - point_dim: Dimensionality of the reconstructed points (e.g., 2 for 2D).
      - dropout: Dropout probability for regularization.
    """
    def __init__(self, n_points, point_dim=2, dropout=0.3):
        # Fix: use the correct class name here.
        super(DeepNetSymm, self).__init__()
        self.n_points = n_points
        self.point_dim = point_dim
        
        # --- Branch 1: Row-wise Processing ---
        # Each row (length n_points) is processed with a shared MLP.
        self.row_embedding = nn.Sequential(
            nn.Linear(n_points, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        # Residual blocks for per-point features.
        self.res_blocks = nn.ModuleList([ResidualBlockSymm(256, dropout) for _ in range(3)])
        
        # Aggregate features over points with 1D convolutions.
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.ln_conv1 = nn.LayerNorm(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.ln_conv2 = nn.LayerNorm(128)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # output shape: (B, channels, 1)
        
        # --- Branch 2: Upper Triangular Processing ---
        # For a symmetric matrix of shape (n_points, n_points), the unique (upper triangular) elements count is:
        n_upper = n_points * (n_points + 1) // 2
        self.upper_embedding = nn.Sequential(
            nn.Linear(n_upper, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # --- Fusion ---
        # Combine the latent vectors from both branches.
        # Branch 1 yields 128 dimensions after convolution/pooling,
        # and Branch 2 yields 256 dimensions.
        self.fc_latent = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # --- Decoder ---
        # Two fully connected layers to reconstruct the coordinates for each point.
        self.fc_dec1 = nn.Linear(256, 256)
        self.ln_dec1 = nn.LayerNorm(256)
        self.fc_dec2 = nn.Linear(256, n_points * point_dim)
        
    def forward(self, x):
        """
        Forward pass.
        Input:
          x: Tensor of shape (B, n_points, n_points) representing the symmetric distance matrix.
        Output:
          Reconstructed coordinates of shape (B, n_points, point_dim)
        """
        # If x is 2D, add a batch dimension.
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Now x becomes (1, n_points, n_points)
        B = x.size(0)
        # ----- Branch 1: Process rows -----
        x_rows = x.view(B * self.n_points, self.n_points)  # each row separately
        row_features = self.row_embedding(x_rows)            # (B*n_points, 256)
        row_features = row_features.view(B, self.n_points, 256)  # (B, n_points, 256)
        
        for block in self.res_blocks:
            row_features = block(row_features)
        
        # Aggregate via convolutions: transpose to (B, channels, n_points)
        conv_input = row_features.transpose(1, 2)
        conv_out = F.silu(self.ln_conv1(self.conv1(conv_input).transpose(1, 2)).transpose(1, 2))
        conv_out = F.silu(self.ln_conv2(self.conv2(conv_out).transpose(1, 2)).transpose(1, 2))
        latent_conv = self.adaptive_pool(conv_out).squeeze(2)  # (B, 128)
        
        # ----- Branch 2: Process upper triangular elements -----
        upper_vectors = []
        for i in range(B):
            mat = x[i]  # shape: (n_points, n_points)
            idx = torch.triu_indices(self.n_points, self.n_points)
            upper_vec = mat[idx[0], idx[1]]  # unique elements
            upper_vectors.append(upper_vec)
        upper_vectors = torch.stack(upper_vectors, dim=0)  # (B, n_upper)
        latent_upper = self.upper_embedding(upper_vectors)  # (B, 256)
        
        # ----- Fusion -----
        latent = torch.cat([latent_conv, latent_upper], dim=1)  # (B, 128+256)
        latent = self.fc_latent(latent)  # (B, 256)
        
        # ----- Decoder: Reconstruct coordinates -----
        dec = F.silu(self.ln_dec1(self.fc_dec1(latent)))
        out = self.fc_dec2(dec)  # (B, n_points * point_dim)
        out = out.view(B, self.n_points, self.point_dim)
        return out





# -------------------------------
# Weighted Stress Loss Function
# -------------------------------
def weighted_stress_loss(pred_points, true_distance, weight_matrix):
    """
    Computes the weighted stress loss between the target distance matrix
    and the pairwise distances computed from the predicted points.
    
    Parameters:
      - pred_points: Tensor of shape (n, 2) with predicted 2D coordinates.
      - true_distance: Tensor of shape (n, n) with target distances.
      - weight_matrix: Tensor of shape (n, n) with weights.
      
    The loss is computed only over the upper triangular matrix (i<j).
    """
    pred_distance = torch.cdist(pred_points, pred_points, p=2)
    mask = torch.triu(torch.ones_like(true_distance), diagonal=1)
    diff = true_distance - pred_distance
    numerator = torch.sum(mask * weight_matrix * (diff ** 2))
    denominator = torch.sum(mask * weight_matrix * (true_distance ** 2))
    loss = torch.sqrt(numerator / denominator)
    return loss

# -------------------------------
# Training Function with tqdm and Save Option
# -------------------------------
def train_deep_mdsnet(distance_matrix_np, weight_matrix_np, n_epochs=1000, lr=1e-3, dropout=0.5,
                      tol=1e-4, patience=50, model_path=None):
    """
    Trains the DeepNet to produce a 2D embedding that minimizes the weighted stress.
    
    Parameters:
      - distance_matrix_np: NumPy array (n x n) with target distances (from compute_distance_matrix).
      - weight_matrix_np: NumPy array (n x n) with weights (absolute correlation values).
      - n_epochs: Maximum number of training epochs.
      - lr: Learning rate.
      - dropout: Dropout rate for the network.
      - tol: Minimum improvement in loss to reset the early stopping counter.
      - patience: Number of epochs to wait for an improvement before stopping early.
      - model_path: Path to a .pth file to load the model from. If None, a new model instance is created.
    
    Returns:
      - pred_points_np: NumPy array (n x 2) of the learned 2D coordinates.
      - model: The trained PyTorch model.
    """
    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}")
    
    n_points = distance_matrix_np.shape[0]
    true_distance = torch.tensor(distance_matrix_np, dtype=torch.float32, device=device)
    weight_matrix = torch.tensor(weight_matrix_np, dtype=torch.float32, device=device)
    
    # Initialize or load the model.
    if model_path is not None:
        model = DeepNet1(n_points, dropout=dropout)
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        model = DeepNetSymm(n_points, dropout=dropout)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Prepare the input tensor (each row is a feature vector for one point).
    input_tensor = torch.tensor(distance_matrix_np, dtype=torch.float32, device=device)
    
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        pred_points = model(input_tensor)  # shape: (n_points, 2)
        loss = weighted_stress_loss(pred_points, true_distance, weight_matrix)
        loss.backward()
        optimizer.step()
        
        # Logging progress.
        if (epoch + 1) % (n_epochs // 10) == 0 or epoch == 0:
            tqdm.tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
        # Early stopping: Check if the loss decreased sufficiently.
        if loss.item() < best_loss - tol:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nBest Loss Value: {best_loss}\t at epoch {epoch-epochs_no_improve}")
            print(f"\nEarly stopping triggered after {epoch+1} epochs with no sufficient improvement.")
            break
    
    # Evaluation: compute the final predictions.
    model.eval()
    with torch.no_grad():
        pred_points = model(input_tensor)
    
    # Move predictions back to CPU.
    pred_points = pred_points.cpu()
    
    # Automatically save the model with a timestamp.
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = f"trained_model_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return pred_points.numpy(), model



# -------------------------------
# Additional Utility: Matrix Statistics
# -------------------------------
def matrix_statistics(matrix):
    """
    Computes and prints overall statistics for a given matrix (excluding trivial values).
    """
    non_zero_elements = matrix[matrix != 0]
    non_one_elements = matrix[matrix != 1]
    non_zero_one_elements = matrix[(matrix != 0) & (matrix != 1)]
    overall_max = np.max(non_one_elements) if non_one_elements.size > 0 else 1
    overall_min = np.min(non_zero_elements) if non_zero_elements.size > 0 else 0
    overall_mean = np.mean(matrix)
    overall_median = np.median(matrix)
    overall_var = np.var(matrix)
    
    non_zero_one_mean = np.mean(non_zero_one_elements)
    non_zero_one_median = np.median(non_zero_one_elements)
    non_zero_one_var = np.var(non_zero_one_elements)
    
    if non_zero_one_elements.size > 0:
        diff_from_median = np.abs(non_zero_one_elements - non_zero_one_median)
        values_far_from_median = diff_from_median > 0.1
        percent_far_from_median = (np.sum(values_far_from_median) / non_zero_one_elements.size) * 100
    else:
        percent_far_from_median = 0
    
    print("Overall Matrix Stats:")
    print("\nContains zeros:      ", np.any(matrix == 0))
    print("Contains ones:         ", np.any(matrix == 1))
    print("  Min value (non-zero):", overall_min)
    print("  Max value (non-one): ", overall_max)
    print("  Mean value:          ", overall_mean)
    print("  Median value:        ", overall_median)
    print("  Variance:            ", overall_var)
    print("\nStats excluding 0s and 1s:")
    print("  Mean:                ", non_zero_one_mean)
    print("  Median:              ", non_zero_one_median) 
    print("  Variance:            ", non_zero_one_var)
    print("  % values >0.1 from median:", f"{percent_far_from_median:.1f}%")
    print("\n\n")


def compute_weighted_stress(original_distance_matrix, embedded_points, weight_matrix):
    """
    Computes the weighted stress between the original distance matrix and the distances
    in the embedding. The stress is defined as the square root of:
    
        (sum_{i<j} w_{ij} (d_{ij} - d̂_{ij})²) / (sum_{i<j} w_{ij} d_{ij}²)
    
    where the weights w_{ij} are defined based on the absolute value of the correlation.
    
    Parameters:
        original_distance_matrix (numpy.ndarray): The target distance matrix.
        embedded_points (numpy.ndarray): The points in the embedding.
        weight_matrix (numpy.ndarray): Matrix of weights (same shape as the distance matrix).
        
    Returns:
        float: The computed weighted stress.
    """
    # Squeeze to 2D if needed since pairwise_distances expects <= 2 dimensions
    points_2d = embedded_points.squeeze(0) if embedded_points.ndim > 2 else embedded_points
    recovered_distance_matrix = pairwise_distances(points_2d)
    diff = original_distance_matrix - recovered_distance_matrix
    numerator = np.sum(weight_matrix * (diff ** 2))
    denominator = np.sum(weight_matrix * (original_distance_matrix ** 2))
    return np.sqrt(numerator / denominator)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Check if CUDA is available and print debug info
    # print("Checking CUDA availability...")
    # print(f"PyTorch version: {torch.__version__}")
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Possible reasons:")
    #     print("1. No NVIDIA GPU detected")
    #     print("2. NVIDIA drivers not installed")
    #     print("3. PyTorch not built with CUDA support")
    #     print("\nTo fix:")
    #     print("- Verify you have an NVIDIA GPU")
    #     print("- Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    #     print("- Install PyTorch with CUDA: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    # else:
    #     print(f"CUDA is available")
    #     print(f"GPU device name: {torch.cuda.get_device_name(0)}")


    #alpha = 1.2067926406393314e-06
    alpha = 0.001
    method = 'linear'
    
    # Load the instance (your JSON file should have the appropriate structure).
    json_path = 'Decision Matrix/Problem setups/C_01.json'
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = load_instance_from_json(data)
    
    # Compute the risk correlation matrix.
    keys, corr_matrix = compute_risk_corr_matrix(instance, alpha)
    weight_matrix_np = np.abs(corr_matrix)  # Use absolute correlations as weights.
    
    # Compute the distance matrix.
    keys, distance_matrix_np = compute_distance_matrix(instance, alpha, method)
    
    print("Starting training of DeepNet...")
    pred_points, model = train_deep_mdsnet(distance_matrix_np, weight_matrix_np, n_epochs=500000, lr=1e-4, dropout=0.2, tol=1e-5, patience=10000, model_path=None)
    pred_points = pred_points.squeeze(0)
    
    #Print the weighted stress:
    weighted_stress = compute_weighted_stress(distance_matrix_np, pred_points, weight_matrix_np)
    print(f"Weighted stress of learned embedding: {weighted_stress:.4f}")

    # Plot the learned 2D embedding.
    plt.figure()
    plt.scatter(pred_points[:, 0], pred_points[:, 1])
    for i, key in enumerate(keys):
        plt.text(pred_points[i, 0], pred_points[i, 1], str(key), fontsize=7)
    plt.title("Learned 2D Embedding via DeepNet")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()



