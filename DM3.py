from my_data_structs import *
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import tqdm

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

class DeepNet(nn.Module):
    def __init__(self, n_points, dropout=0.3):
        """
        A deep neural network with 8 layers that gradually reduces
        the dimensionality from n_points to 2.
        
        Parameters:
          - n_points: The input dimension.
          - dropout: Dropout rate for regularization.
        """
        super(DeepNet, self).__init__()
        
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
    loss = numerator / denominator
    return loss

# -------------------------------
# Training Function with tqdm and Save Option
# -------------------------------
def train_deep_mdsnet(distance_matrix_np, weight_matrix_np, n_epochs=1000, lr=1e-3, dropout=0.5):
    """
    Trains the DeepNet to produce a 2D embedding that minimizes the weighted stress.
    
    Parameters:
      - distance_matrix_np: NumPy array (n x n) with target distances (from compute_distance_matrix).
      - weight_matrix_np: NumPy array (n x n) with weights (absolute correlation values).
      - n_epochs: Number of training epochs.
      - lr: Learning rate.
      - dropout: Dropout rate for the network.
    
    Returns:
      - pred_points_np: NumPy array (n x 2) of the learned 2D coordinates.
      - model: The trained PyTorch model.
    """
    n_points = distance_matrix_np.shape[0]
    true_distance = torch.tensor(distance_matrix_np, dtype=torch.float32)
    weight_matrix = torch.tensor(weight_matrix_np, dtype=torch.float32)
    
    # Initialize the model and optimizer.
    model = DeepNet1(n_points, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # In this setup, the input is the distance matrix itself (each row is a feature vector for one point).
    input_tensor = torch.tensor(distance_matrix_np, dtype=torch.float32)
    
    model.train()
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        pred_points = model(input_tensor)  # shape: (n_points, 2)
        loss = weighted_stress_loss(pred_points, true_distance, weight_matrix)
        loss.backward()
        optimizer.step()
        
        # Optionally log progress.
        if (epoch + 1) % (n_epochs // 10) == 0 or epoch == 0:
            tqdm.tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    # Evaluation: compute the final predictions.
    model.eval()
    with torch.no_grad():
        pred_points = model(input_tensor)
    
    # Ask user whether to save the trained model.
    save_model = input("Save the trained model? (y/n): ").strip().lower()
    if save_model == 'y':
        save_path = input("Enter the file path to save the model (e.g., model.pth): ").strip()
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
    recovered_distance_matrix = pairwise_distances(embedded_points)
    diff = original_distance_matrix - recovered_distance_matrix
    numerator = np.sum(weight_matrix * (diff ** 2))
    denominator = np.sum(weight_matrix * (original_distance_matrix ** 2))
    return np.sqrt(numerator / denominator)

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    alpha = 0.05
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
    pred_points, model = train_deep_mdsnet(distance_matrix_np, weight_matrix_np,
                                           n_epochs=20000, lr=1e-5, dropout=0.2)
    
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



