from my_data_structs import *
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tqdm

import os
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.4.2"

# Import rpy2 modules
import rpy2.robjects as ro
print(ro.r("version"))
from rpy2.robjects import numpy2ri
# Activate automatic conversion between numpy and R objects
numpy2ri.activate()

# Load the R package "smacof"
ro.r('library(smacof)')

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

def compute_distance_matrix(instance, alpha=1e-15, method="linear"):
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
# Weighted MDS using R's SMACOF Package via rpy2
# -------------------------------
def recover_points_MDS_weighted(distance_matrix, weight_matrix, n_dimensions=2):
    """
    Uses R's SMACOF routine (via rpy2 and the R package "smacof") to perform MDS while minimizing
    a weighted stress function. The weight matrix is applied so that pairs with higher absolute
    correlations are given more importance.
    
    Parameters:
        distance_matrix (numpy.ndarray): Precomputed distance matrix (dissimilarities).
        weight_matrix (numpy.ndarray): Weight matrix (typically, absolute values of the correlations).
        n_dimensions (int): Number of dimensions for the embedding.
    
    Returns:
        tuple:
            - points (numpy.ndarray): The configuration (embedding) as an array of shape (n_points, n_dimensions).
            - weighted_stress (float): The weighted stress value achieved by the SMACOF algorithm.
    """
    # Convert distance and weight matrices to R objects and assign them to the global environment.
    ro.globalenv["r_distance"] = ro.conversion.py2rpy(distance_matrix)
    ro.globalenv["r_weight"] = ro.conversion.py2rpy(weight_matrix)
    
    # Convert the full distance matrix to an R "dist" object using as.dist.
    ro.r('r_distance <- as.dist(r_distance)')
    
    # Call the smacofSym function in R with the distance and weight matrices.
    # This R call performs the weighted MDS with the desired number of dimensions.
    r_code = f"""
    result <- smacofSym(delta = r_distance, weightmat = r_weight, ndim = {n_dimensions})
    """
    ro.r(r_code)
    
    # Retrieve the configuration (embedding) and the weighted stress from the result.
    conf = np.array(ro.r('result$conf'))
    weighted_stress = float(ro.r('result$stress'))
    
    return conf, weighted_stress


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
# Testing Function using Weighted MDS (via R's SMACOF)
# -------------------------------
def test_embeddings(instance, alpha_values, distance_methods, plot=True, top_n=5, mat_stats=False):
    """
    Tests weighted MDS embeddings over different α values and distance transformation methods.
    
    For each configuration, the function computes:
      - The normalized correlation and distance matrices.
      - The weight matrix (absolute values of the correlations).
      - A weighted MDS embedding using R's SMACOF routine via rpy2.
    
    The weighted stress is obtained directly from the R SMACOF output.
    
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
        # Compute correlation matrix and weight matrix (absolute correlations)
        keys, corr_matrix = compute_risk_corr_matrix(instance, alpha=alpha)
        weight_matrix = np.abs(corr_matrix)
        
        for d_method in tqdm.tqdm(distance_methods, desc=f"Distance methods (α={alpha})", leave=False):
            if plot:
                print(f"\n=== Testing for α = {alpha}, distance transformation = {d_method} ===")
            keys, dist_matrix = compute_distance_matrix(instance, alpha=alpha, method=d_method)
            
            if mat_stats:
                print(f"\nCORRELATION MATRIX (α={alpha}):")
                matrix_statistics(corr_matrix)
                print(f"\nDISTANCE MATRIX (α={alpha}, method={d_method}):")
                matrix_statistics(dist_matrix)
            
            # --- Weighted MDS Embedding using R's SMACOF ---
            if plot:
                print(f"\nEmbedding method: Weighted MDS (via R's SMACOF)")
            points, weighted_stress = recover_points_MDS_weighted(dist_matrix, weight_matrix, n_dimensions=2)
            if plot:
                print(f"Weighted Stress (SMACOF): {weighted_stress:.4f}")
                title = f"Weighted MDS (α={alpha}, d_method={d_method})"
                plot_embedding(points, title, keys)
            results.append({
                "alpha": alpha,
                "distance_method": d_method,
                "embedding": "Weighted MDS (SMACOF)",
                "weighted_stress": weighted_stress
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
    #alpha_values = np.linspace(0.0004119473684210526, 0.0004457894736842105, 30)
    alpha_values = np.logspace(-20, 3, 50)
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
    
    # Run the testing function using weighted MDS (via R's SMACOF)
    results = test_embeddings(instance, alpha_values, distance_methods, plot=False, top_n=5, mat_stats=True)
