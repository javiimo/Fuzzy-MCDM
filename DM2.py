from my_data_structs import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.metrics import pairwise_distances
from umap import UMAP
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# -------------------------------
# Risk Correlation and Distance Computations
# -------------------------------
def compute_risk_corr_np(risk1, risk2, alpha=1e-15):
    """
    Computes a modified correlation between two risk series that captures jump sizes.
    """
    diff1 = np.diff(risk1)
    diff2 = np.diff(risk2)
    similarity = np.exp(-alpha * np.abs(diff1 - diff2))  # values in (0,1]
    sign = np.where(diff1 * diff2 >= 0, 1, -1)
    contributions = sign * similarity
    return np.sum(contributions)

def compute_risk_corr_matrix(instance, alpha=1e-15):
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
        method (str): Transformation method. Options:
            - "linear":     d = (1 - r) / 2
            - "sqrt":       d = sqrt((1 - r) / 2)
            - "arccos":     d = arccos(r) / π
            - "logistic":   d = (f(r) - f(1)) / (f(-1) - f(1))
                            where f(r)=1/(1+exp(a*r)) with a default steepness a=2.
            - "exponential":d = (exp(-k*r) - exp(-k)) / (exp(k) - exp(-k)) with k=1.
            - "power2":     d = ((1 - r)/2)^2   (a power > 1 compresses high r values)
            - "power1/3":   d = ((1 - r)/2)^(1/3)  (a power < 1 spreads out mid values)
            - "arctan":     d = arctan(alpha*(1 - r)) / arctan(2*alpha) with alpha=1.
            - "sine":       d = sin((π/4)*(1 - r))
            
    Returns:
        numpy.ndarray: A distance matrix with values in [0, 1].

    Raises:
        ValueError: If any correlation is outside [-1, 1] or if an unknown method is given.
    """
    # Convert input to a NumPy array
    corr_matrix = np.array(corr_matrix)
    
    # Check that all correlations lie in [-1, 1]
    if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
        raise ValueError("All correlation values must be in the range [-1, 1].")
    
    if method == "linear":
        distance_matrix = (1 - corr_matrix) / 2
    elif method == "sqrt":
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    elif method == "arccos":
        distance_matrix = np.arccos(corr_matrix) / np.pi
    elif method == "logistic":
        # Logistic mapping with steepness parameter a.
        a = 2
        f = lambda r: 1 / (1 + np.exp(a * r))
        f1 = f(1)      # Minimum value at r=1
        f_neg1 = f(-1) # Maximum value at r=-1
        distance_matrix = (f(corr_matrix) - f1) / (f_neg1 - f1)
    elif method == "exponential":
        # Exponential mapping with parameter k.
        k = 1
        distance_matrix = (np.exp(-k * corr_matrix) - np.exp(-k)) / (np.exp(k) - np.exp(-k))
    elif method == "power2":
        # A power > 1 compresses differences at high correlations.
        distance_matrix = ((1 - corr_matrix) / 2)**2
    elif method == "power1/3":
        # A power < 1 spreads out the values around 0, making near-zero correlations appear farther.
        distance_matrix = ((1 - corr_matrix) / 2)**(1/3)
    elif method == "arctan":
        # Arctan mapping with scaling parameter alpha.
        alpha = 1
        distance_matrix = np.arctan(alpha * (1 - corr_matrix)) / np.arctan(alpha * 2)
    elif method == "sine":
        # Sine mapping: note that sin(π/4)=~0.7071, so r=0 maps to ~0.7071.
        distance_matrix = np.sin((np.pi/4) * (1 - corr_matrix))
    else:
        raise ValueError(f"Unknown distance transformation method: {method}")
    
    return distance_matrix

def compute_distance_matrix(instance, alpha=1e-15, method="linear"):
    """
    Computes a distance matrix from the normalized correlation matrix.
    After transforming, distances are normalized (except the diagonal).
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
# Embedding Functions
# -------------------------------
def recover_points_MDS(distance_matrix, n_dimensions=2):
    mds = MDS(n_components=n_dimensions, dissimilarity="precomputed", random_state=42)
    points = mds.fit_transform(distance_matrix)
    return points, mds.stress_

def recover_points_Isomap(distance_matrix, n_dimensions=2):
    isomap = Isomap(n_components=n_dimensions, metric="precomputed")
    points = isomap.fit_transform(distance_matrix)
    return points

def recover_points_TSNE(distance_matrix, n_dimensions=2):
    # Set init to "random" to avoid the error with metric="precomputed"
    tsne = TSNE(n_components=n_dimensions, metric="precomputed", random_state=42, init="random")
    points = tsne.fit_transform(distance_matrix)
    return points

def recover_points_UMAP(distance_matrix, n_dimensions=2):
    umap_embedder = UMAP(n_components=n_dimensions, metric="precomputed", random_state=42, verbose=False)
    points = umap_embedder.fit_transform(distance_matrix)
    return points

def plot_embedding(points, title, keys=None):
    """
    Plots the 2D embedding. Optionally labels each point with its intervention key.
    """
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    if keys is not None:
        for i, key in enumerate(keys):
            plt.text(points[i, 0], points[i, 1], str(key), fontsize=9)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# -------------------------------
# High-dimensional Embedding + PCA Projection
# -------------------------------
def high_dim_embedding_from_distance(dist_matrix, method="UMAP", high_dim=5):
    """
    Computes a high-dimensional embedding (of dimension high_dim) from a precomputed distance matrix
    using one of the methods UMAP, Isomap, TSNE, or MDS.
    """
    if method == "UMAP":
        embedder = UMAP(n_components=high_dim, metric="precomputed", random_state=42, verbose=False)
    elif method == "Isomap":
        embedder = Isomap(n_components=high_dim, metric="precomputed")
    elif method == "TSNE":
        # Override default init to 'random' because "pca" is incompatible with precomputed distances.
        embedder = TSNE(n_components=high_dim, metric="precomputed", random_state=42, init="random")
    elif method == "MDS":
        embedder = MDS(n_components=high_dim, dissimilarity="precomputed", random_state=42)
    else:
        raise ValueError("Unknown method for high-dimensional embedding")
    high_dim_points = embedder.fit_transform(dist_matrix)
    return high_dim_points

def project_to_pca_plane(points, target_dim=2):
    """
    Performs PCA on the high-dimensional points and returns their projection on the best-fit target_dim plane.
    This is equivalent to an orthogonal projection onto the linear subspace that best approximates the data.
    """
    pca = PCA(n_components=target_dim)
    projected_points = pca.fit_transform(points)
    return projected_points

# -------------------------------
# Testing Function (with PCA option)
# -------------------------------
def test_embeddings(instance, alpha_values, distance_methods, map_methods, high_dims=None, plot=True, top_n=5, mat_stats = False):
    """
    Loops over different α values and distance transformation methods,
    and applies various embedding methods.
    
    For each case, prints the weighted stress and (if applicable) stress (for MDS) and plots the embedding.
    If an embedding method in map_methods starts with "PCA_", then it uses the corresponding
    high-dimensional embedding method followed by an orthogonal projection (via PCA) to 2D.
    
    Parameters:
      - instance: Data instance with risk time series.
      - alpha_values: List of α values to test.
      - distance_methods: List of methods for transforming correlations into distances.
      - map_methods: List of embedding method names to test.
                     Valid names for standard methods: "MDS", "Isomap", "TSNE", "UMAP".
                     For PCA-based methods, use names like "PCA_UMAP", "PCA_Isomap", etc.
      - high_dims: (Optional) List of high-dimensional target values for the PCA-based approach.
                   For example, [3, 5, 10]. If None, a default list [3, 5] is used.
      - plot: Boolean flag to control whether to show plots (default: True)
      - top_n: Number of best performing methods to display when plot=False (default: 5)
    """
    # Default high_dims if not provided
    if high_dims is None:
        high_dims = [3, 5]
    
    # Standard embedding methods dictionary
    standard_methods = {
        "MDS": recover_points_MDS,
        "Isomap": recover_points_Isomap,
        "TSNE": recover_points_TSNE,
        "UMAP": recover_points_UMAP
    }
    
    results = []
    
    for alpha in tqdm.tqdm(alpha_values, desc="Alpha values"):
        for d_method in tqdm.tqdm(distance_methods, desc=f"Distance methods (α={alpha})", leave=False):
            if plot:
                print(f"\n=== Testing for α = {alpha}, distance transformation = {d_method} ===")
            keys, corr_matrix = compute_risk_corr_matrix(instance, alpha=alpha)
            keys, dist_matrix = compute_distance_matrix(instance, alpha=alpha, method=d_method)
            if mat_stats:
                print(f"\nCORRELATION MATRIX (α={alpha}):")
                matrix_statistics(corr_matrix)
                print(f"\nDISTANCE MATRIX (α={alpha}, method={d_method}):")
                matrix_statistics(dist_matrix)
            
            for emb_name in tqdm.tqdm(map_methods, desc=f"Embedding methods (α={alpha}, d={d_method})", leave=False):
                # Check if we're using the PCA-based hybrid method
                if emb_name.startswith("PCA_"):
                    # Extract the underlying method name, e.g., "PCA_UMAP" -> "UMAP"
                    underlying_method = emb_name.split("_")[1]
                    for high_dim in high_dims:
                        if plot:
                            print(f"\nEmbedding method: {emb_name} with high_dim = {high_dim}")
                        # Get high-dimensional embedding from the precomputed distance matrix
                        high_dim_points = high_dim_embedding_from_distance(dist_matrix, method=underlying_method, high_dim=high_dim)
                        # Project onto the best-fit 2D plane via PCA (orthogonal projection)
                        points = project_to_pca_plane(high_dim_points, target_dim=2)
                        # Compute quality metric using weighted stress
                        stress_val = compute_weighted_stress(dist_matrix, points, np.abs(corr_matrix))
                        if plot:
                            print(f"Weighted Stress: {stress_val:.4f}")
                            title = f"{emb_name} (α={alpha}, d_method={d_method}, high_dim={high_dim})"
                            plot_embedding(points, title, keys)
                        results.append({
                            "alpha": alpha,
                            "distance_method": d_method,
                            "embedding": emb_name,
                            "high_dim": high_dim,
                            "rmse": stress_val,
                            "rmse_percentage": None,
                            "stress": stress_val
                        })
                else:
                    # Standard embedding method
                    if plot:
                        print(f"\nEmbedding method: {emb_name}")
                    emb_func = standard_methods.get(emb_name)
                    if emb_func is None:
                        raise ValueError(f"Embedding method {emb_name} not recognized.")
                    if emb_name == "MDS":
                        points, stress = emb_func(dist_matrix, n_dimensions=2)
                        if plot:
                            print("MDS Stress:", stress)
                    else:
                        points = emb_func(dist_matrix, n_dimensions=2)
                        stress = None
                    # Compute quality metric using weighted stress
                    stress_val = compute_weighted_stress(dist_matrix, points, np.abs(corr_matrix))
                    if plot:
                        print(f"Weighted Stress: {stress_val:.4f}")
                        title = f"{emb_name} (α={alpha}, d_method={d_method})"
                        plot_embedding(points, title, keys)
                    results.append({
                        "alpha": alpha,
                        "distance_method": d_method,
                        "embedding": emb_name,
                        "rmse": stress_val,
                        "rmse_percentage": None,
                        "stress": stress_val
                    })

    if not plot:
        # Sort results by weighted stress (stored under "rmse")
        sorted_results = sorted(results, key=lambda x: x['rmse'])
        print("\n=== TOP", top_n, "PERFORMING METHODS ===")
        print("\nRanked by Weighted Stress:")
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. Configuration:")
            print(f"   Embedding Method: {result['embedding']}")
            if 'high_dim' in result and result['high_dim'] is not None:
                print(f"   High Dimensions: {result['high_dim']}")
            print(f"   Distance Method: {result['distance_method']}")
            print(f"   Alpha: {result['alpha']}")
            print(f"   Weighted Stress: {result['rmse']:.4f}")
            if result['stress'] is not None:
                print(f"   Stress: {result['stress']:.4f}")
            
    return results

# -------------------------------
# Example Usage
# -------------------------------
def matrix_statistics(matrix):
    # Compute overall statistics (exclude zeros and ones)
    non_zero_elements = matrix[matrix != 0]
    non_one_elements = matrix[matrix != 1]
    non_zero_one_elements = matrix[(matrix != 0) & (matrix != 1)]
    # Calculate overall statistics
    overall_max = np.max(non_one_elements) if non_one_elements.size > 0 else 1
    overall_min = np.min(non_zero_elements) if non_zero_elements.size > 0 else 0
    overall_mean = np.mean(matrix)
    overall_median = np.median(matrix)
    overall_var = np.var(matrix)
    
    # Calculate statistics excluding 0s and 1s
    non_zero_one_mean = np.mean(non_zero_one_elements)
    non_zero_one_median = np.median(non_zero_one_elements)
    non_zero_one_var = np.var(non_zero_one_elements)
    
    # Calculate percentage of values more than 0.1 from median
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

if __name__ == "__main__":
    json_path = 'Decision Matrix\Problem setups\C_01.json'
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = load_instance_from_json(data)
    keys, corr_matrix = compute_risk_corr_matrix(instance)
    keys, distance_matrix = compute_distance_matrix(instance)
    print("Intervention Keys:", keys)
    print("Distance Matrix:\n", distance_matrix)

    ####################################################
    #         Statistics of the distance matrix
    ####################################################
    print("CORRELATION MATRIX:")
    matrix_statistics(corr_matrix)
    print("DISTANCE MATRIX:")
    matrix_statistics(distance_matrix)
    
    ####################################################
    #         Testing map methods
    ####################################################
    
    # Define the α values and distance transformation methods you want to test:
    #alpha_values = np.logspace(-20, 3, 50)
    #alpha_values = np.linspace(0.0004119473684210526, 0.0004457894736842105, 30)
    alpha_values = [0.0023299518105153816]
    distance_methods = [
                        #"linear", 
                        #"sqrt", 
                        #"arccos",
                        #"logistic",
                        #"exponential", 
                        #"power2",
                        #"power1/3",
                        "arctan",
                        #"sine"
                        ]
    map_methods = [
                    #"MDS",
                    "Isomap",
                    #"TSNE",
                    #"UMAP",
                    #"PCA_UMAP", 
                    #"PCA_Isomap", 
                    #"PCA_TSNE"
                    ]

    high_dims = [
                15,
                20,
                25,
                30,
                35]
    
    # Run tests (uncomment the following line once your instance is defined)
    results = test_embeddings(instance, alpha_values, distance_methods, map_methods, high_dims, plot=True, top_n=5, mat_stats=True)
