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
# Embedding Function (MDS Only)
# -------------------------------
def recover_points_MDS(distance_matrix, n_dimensions=2):
    """
    Uses Multidimensional Scaling (MDS) to embed points given a precomputed distance matrix.
    
    Parameters:
        distance_matrix (numpy.ndarray): Precomputed distance matrix.
        n_dimensions (int): Number of dimensions for the embedding.
        
    Returns:
        points (numpy.ndarray): Embedded points.
        mds_stress (float): Raw stress value from the MDS algorithm (for reference).
    """
    mds = MDS(n_components=n_dimensions, dissimilarity="precomputed", random_state=42)
    points = mds.fit_transform(distance_matrix)
    return points, mds.stress_


def plot_embedding(points, title, keys=None):
    """
    Plots the 2D embedding and optionally labels each point with its key.
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
# Testing Function using Weighted Stress
# -------------------------------
def test_embeddings(instance, alpha_values, distance_methods, plot=True, top_n=5, mat_stats=False):
    """
    Tests MDS embeddings over different α values and distance transformation methods.
    
    For each configuration, the function computes:
      - The normalized correlation and distance matrices.
      - An MDS embedding.
      - A weighted stress measure computed using weights based on the absolute value
        of the normalized correlations.
    
    The weighted stress is defined as:
      sqrt( sum_{i<j} w_{ij}(d_{ij} - d̂_{ij})² / sum_{i<j} w_{ij} d_{ij}² )
    where w_{ij} = |corr_{ij}|.
    
    Parameters:
      - instance: Data instance with risk time series.
      - alpha_values: List of α values to test.
      - distance_methods: List of methods for transforming correlations into distances.
      - plot: Boolean flag to control whether to show plots (default: True)
      - top_n: Number of best performing configurations to display when plot=False (default: 5)
      - mat_stats: If True, prints statistics for the correlation and distance matrices.
      
    Returns:
      - results (list): A list of dictionaries containing the configuration and performance metrics.
    """
    results = []
    
    for alpha in tqdm.tqdm(alpha_values, desc="Alpha values"):
        # Compute correlation matrix and corresponding weight matrix from correlations.
        keys, corr_matrix = compute_risk_corr_matrix(instance, alpha=alpha)
        weight_matrix = np.abs(corr_matrix)  # weights based on absolute correlation
        
        for d_method in tqdm.tqdm(distance_methods, desc=f"Distance methods (α={alpha})", leave=False):
            if plot:
                print(f"\n=== Testing for α = {alpha}, distance transformation = {d_method} ===")
            keys, dist_matrix = compute_distance_matrix(instance, alpha=alpha, method=d_method)
            
            if mat_stats:
                print(f"\nCORRELATION MATRIX (α={alpha}):")
                matrix_statistics(corr_matrix)
                print(f"\nDISTANCE MATRIX (α={alpha}, method={d_method}):")
                matrix_statistics(dist_matrix)
            
            # --- MDS Embedding ---
            if plot:
                print(f"\nEmbedding method: MDS")
            points, mds_raw_stress = recover_points_MDS(dist_matrix, n_dimensions=2)
            # Compute the weighted stress using our custom function.
            w_stress = compute_weighted_stress(dist_matrix, points, weight_matrix)
            if plot:
                print(f"MDS Raw Stress (from algorithm): {mds_raw_stress:.4f}")
                print(f"Weighted Stress: {w_stress:.4f}")
                title = f"MDS (α={alpha}, d_method={d_method})"
                plot_embedding(points, title, keys)
            results.append({
                "alpha": alpha,
                "distance_method": d_method,
                "embedding": "MDS",
                "weighted_stress": w_stress,
                "raw_stress": mds_raw_stress
            })
    
    if not plot:
        # Sort results by weighted stress (lower is better)
        sorted_results = sorted(results, key=lambda x: x['weighted_stress'])
        print("\n=== TOP", top_n, "PERFORMING CONFIGURATIONS ===")
        print("\nRanked by weighted stress:")
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. Configuration:")
            print(f"   Embedding Method: {result['embedding']}")
            print(f"   Distance Method: {result['distance_method']}")
            print(f"   Alpha: {result['alpha']}")
            print(f"   Weighted Stress: {result['weighted_stress']:.4f}")
            print(f"   MDS Raw Stress: {result['raw_stress']:.4f}")
            
    return results


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


# -------------------------------
# Main Execution Example
# -------------------------------
if __name__ == "__main__":
    json_path = 'Decision Matrix/Problem setups/C_01.json'
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = load_instance_from_json(data)
    keys, corr_matrix = compute_risk_corr_matrix(instance)
    keys, distance_matrix = compute_distance_matrix(instance)
    print("Intervention Keys:", keys)
    print("Distance Matrix:\n", distance_matrix)

    # Display statistics of the correlation and distance matrices
    print("CORRELATION MATRIX:")
    matrix_statistics(corr_matrix)
    print("DISTANCE MATRIX:")
    matrix_statistics(distance_matrix)
    
    # Define the α values and distance transformation methods to test
    alpha_values = np.linspace(0.0004119473684210526, 0.0004457894736842105, 30)
    distance_methods = [
        "linear", 
        "sqrt", 
        "arccos",
        "logistic",
        "exponential", 
        "power2",
        "power1/3",
        "arctan",
        "sine"
    ]
    map_methods = ["MDS"]  # Only MDS is used
    
    # Run the testing function using weighted stress as the quality metric
    results = test_embeddings(instance, alpha_values, distance_methods, plot=False, top_n=5, mat_stats=True)