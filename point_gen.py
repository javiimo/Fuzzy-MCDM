from my_data_structs import *
import json
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime
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
def compute_risk_corr_np(risk1, risk2):
    """
    Computes a modified correlation between two risk series based solely on the sign of their differences.
    
    For each pair of consecutive differences:
      - Adds 1 if both differences are 0 or both are nonzero and have the same sign.
      - Subtracts 1 if both differences are nonzero and have opposite signs.
      - Leaves it as 0 if only one of the differences is 0.
    """
    diff1 = np.diff(risk1)
    diff2 = np.diff(risk2)
    
    contributions = np.zeros_like(diff1, dtype=float)
    
    # Condition: both differences are 0 or both nonzero and have the same sign
    condition_positive = ((diff1 > 0) & (diff2 > 0)) | ((diff1 < 0) & (diff2 < 0)) | ((diff1 == 0) & (diff2 == 0))
    contributions[condition_positive] = 1
    
    # Condition: both differences are nonzero and have opposite signs
    condition_negative = (diff1 * diff2 < 0)
    contributions[condition_negative] = -1
    
    # Cases where one diff is 0 and the other nonzero remain 0
    return np.sum(contributions)

def compute_risk_corr_matrix(instance):
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
            corr = compute_risk_corr_np(risk_i, risk_j)
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
        method (str): Transformation method.
        
    Returns:
        numpy.ndarray: A distance matrix with values in [0, 1].
    """
    corr_matrix = np.array(corr_matrix)
    
    # Validate correlation range.
    if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
        raise ValueError("All correlation values must be in the range [-1, 1].")
    
    if method == "linear":
        distance_matrix = (1 - corr_matrix)/2
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

def compute_distance_matrix(instance, method="linear"):
    """
    Computes a distance matrix from the normalized correlation matrix.
    For off-diagonal entries, if the correlation (from norm_corr_matrix) is ≤ 0, the distance is set to NaN.
    Otherwise, valid distances are rescaled to the range [epsilon, 1].
    """
    keys, norm_corr_matrix = compute_risk_corr_matrix(instance)
    distance_matrix = transform_corr_to_distance(norm_corr_matrix, method=method)
    
    off_diag_mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    # Only consider valid (positive) correlations for rescaling.
    #valid_mask = off_diag_mask & (norm_corr_matrix > 0)
    # if np.any(valid_mask):
    #     d_min = np.nanmin(distance_matrix[valid_mask])
    #     d_max = np.nanmax(distance_matrix[valid_mask])
    #     epsilon = 1e-6
    #     distance_matrix[valid_mask] = epsilon + (1 - epsilon) * (distance_matrix[valid_mask] - d_min) / (d_max - d_min)
    
    # Set distances to NaN where correlation is ≤ 0 (except on the diagonal)
    invalid_mask = off_diag_mask & (norm_corr_matrix <= 0)
    distance_matrix[invalid_mask] = np.nan
    return keys, distance_matrix

# -------------------------------
# Non-metric MDS using scikit-learn
# -------------------------------
def recover_points_MDS_weighted(distance_matrix, weight_matrix, n_dimensions=2, random_state=42):
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
    ro.r(f'set.seed({random_state})')

    
    # Convert the full distance matrix to an R "dist" object using as.dist.
    ro.r('r_distance <- as.dist(r_distance)')
    
    # Call the smacofSym function in R with type = "ordinal" to perform non-metric MDS.
    r_code = f"""
    result <- smacofSym(delta = r_distance, weightmat = r_weight, ndim = {n_dimensions}, type = "ordinal", init = "random", verbose = TRUE, itmax = 500, modulus = 1)
    """
    ro.r(r_code)
    
    # Retrieve the configuration (embedding) and the weighted stress from the result.
    points = np.array(ro.r('result$conf'))
    weighted_stress = ro.r('result$stress')[0]  # Extract the first element to avoid deprecation warning
    
    return points, weighted_stress

# -------------------------------
# Plotting Function with Integrated Draw Lines and Network Stats
# -------------------------------
def plot_embedding(points, title, keys=None, distance_matrix=None, weight_matrix=None, draw_lines=False):
    """
    Plots the 2D embedding, optionally labels points, draws lines to the highest weighted neighbor,
    and prints network connection statistics.
    
    Parameters:
        points (numpy.ndarray): Array of shape (n_points, 2) with the embedding coordinates.
        title (str): Plot title.
        keys (list, optional): List of labels for each point.
        distance_matrix (numpy.ndarray, optional): Distance matrix used to compute connectivity statistics.
        weight_matrix (numpy.ndarray, optional): Weight matrix (e.g., absolute correlations) used for drawing lines.
        draw_lines (bool): If True, draws a line from each node to its highest weighted connection.
    """
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    if keys is not None:
        for i, key in enumerate(keys):
            plt.text(points[i, 0], points[i, 1], str(key), fontsize=9)
    
    # Draw highest weighted connection lines if requested
    if draw_lines and (weight_matrix is not None):
        num_points = len(points)
        drawn_edges = set()
        for i in range(num_points):
            row = weight_matrix[i].copy()
            row[i] = -1  # Exclude self
            j = np.argmax(row)
            max_weight = weight_matrix[i, j]
            if max_weight > 0:
                edge = (min(i, j), max(i, j))
                if edge not in drawn_edges:
                    drawn_edges.add(edge)
                    x_coords = [points[i, 0], points[j, 0]]
                    y_coords = [points[i, 1], points[j, 1]]
                    plt.plot(x_coords, y_coords, color='red', alpha=max_weight, linewidth=1)
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # If distance matrix is provided, compute connectivity stats.
    if distance_matrix is not None:
        connection_counts = []
        num_points = distance_matrix.shape[0]
        for i in range(num_points):
            # Count non-NaN connections, excluding self.
            count = np.sum(~np.isnan(distance_matrix[i])) - 1
            connection_counts.append(count)
        connection_counts = np.array(connection_counts)
        min_conn = connection_counts.min()
        max_conn = connection_counts.max()
        median_conn = np.median(connection_counts)
        std_conn = connection_counts.std()
        # Percentage of nodes whose connection count is outside one sigma from the median.
        outside = np.sum((connection_counts < (median_conn - std_conn)) | (connection_counts > (median_conn + std_conn)))
        perc_outside = 100 * outside / len(connection_counts)
        print("Connection Stats:")
        print("  Min connections:         ", min_conn)
        print("  Max connections:         ", max_conn)
        print("  Median connections:      ", median_conn)
        print("  % outside one sigma:     {:.1f}%".format(perc_outside))
    
    plt.show()

# -------------------------------
# Testing Function using Non-metric MDS (scikit-learn)
# -------------------------------
def test_embeddings(instance, distance_methods, plot=True, top_n=5, mat_stats=False, draw_lines=False):
    """
    Tests non-metric MDS embeddings over different α values and distance transformation methods.
    
    For each configuration, the function computes:
      - The normalized correlation and distance matrices.
      - The weight matrix (absolute correlations).
      - A non-metric MDS embedding using scikit-learn.
    
    The stress is obtained directly from the scikit-learn MDS.
    
    Parameters:
      - instance: Data instance with risk time series.
      - distance_methods: List of methods for transforming correlations into distances.
      - plot: Boolean flag to control whether to show plots (default: True)
      - top_n: Number of best performing configurations to display when plot=False (default: 5)
      - mat_stats: If True, prints statistics for the correlation and distance matrices.
      - draw_lines: If True, draws highest weighted lines and prints connectivity stats.
      
    Returns:
      - results (list): A list of dictionaries containing the configuration and performance metrics.
    """
    results = []
    
    # Compute correlation matrix and weight matrix (absolute correlations)
    keys, corr_matrix = compute_risk_corr_matrix(instance)
    weight_matrix = np.where(corr_matrix <= 0, 0, corr_matrix)
    
    for d_method in tqdm.tqdm(distance_methods, desc=f"Distance methods", leave=False):
        print(f"\n=== Testing for distance transformation = {d_method} ===")
        keys, dist_matrix = compute_distance_matrix(instance, method=d_method)
        
        if mat_stats:
            print(f"\nCORRELATION MATRIX:")
            matrix_statistics(corr_matrix)
            print(f"\nDISTANCE MATRIX, method={d_method}):")
            matrix_statistics(dist_matrix)
        
        # --- Non-metric MDS Embedding using scikit-learn ---
        print(f"\nEmbedding method: Non-metric MDS (scikit-learn)")
        points, stress = recover_points_MDS_weighted(distance_matrix=dist_matrix, weight_matrix=weight_matrix,n_dimensions=2)
        print(f"Stress (Non-metric MDS): {stress:.4f}")
        title = f"Non-metric MDS, d_method={d_method})"
        plot_embedding(points, title, keys, distance_matrix=dist_matrix, weight_matrix=weight_matrix, draw_lines=draw_lines)
        
        results.append({
            "distance_method": d_method,
            "embedding": "Non-metric MDS (scikit-learn)",
            "stress": stress
        })
    
    if not plot:
        # Sort results by stress (lower is better)
        sorted_results = sorted(results, key=lambda x: x['stress'])
        print("\n=== TOP", top_n, "PERFORMING CONFIGURATIONS ===")
        print("\nRanked by stress:")
        for i, result in enumerate(sorted_results[:top_n], 1):
            print(f"\n{i}. Configuration:")
            print(f"   Embedding Method: {result['embedding']}")
            print(f"   Distance Method: {result['distance_method']}")
            print(f"   Stress: {result['stress']:.4f}")
            
    return results

def compute_and_save_embedding(instance, distance_method, points_file, mat_stats=False):
    """
    Compute non-metric MDS embedding for a given distance method, plot the embedding,
    print matrix statistics and stress, and save the computed points to a file.

    Parameters:
      instance: Data instance with risk time series.
      distance_method: Method for transforming correlations into distances.
      points_file: File path where the embedding points will be saved.
      mat_stats: If True, prints statistics for the correlation and distance matrices (default: False).

    Returns:
      points: The computed embedding points.
      stress: The stress of the embedding.
      keys: List of intervention names corresponding to each point.
    """
    # Compute correlation matrix and weight matrix (absolute correlations)
    keys, corr_matrix = compute_risk_corr_matrix(instance)
    weight_matrix = np.where(corr_matrix <= 0, 0, corr_matrix)
    
    # Compute distance matrix using the provided distance method
    keys, dist_matrix = compute_distance_matrix(instance, method=distance_method)
    
    if mat_stats:
        print(f"\nCORRELATION MATRIX:")
        matrix_statistics(corr_matrix)
        print(f"\nDISTANCE MATRIX (method={distance_method}):")
        matrix_statistics(dist_matrix)
    
    # --- Non-metric MDS Embedding using scikit-learn ---
    print(f"\nEmbedding method: Non-metric MDS (scikit-learn)")
    points, stress = recover_points_MDS_weighted(distance_matrix=dist_matrix, weight_matrix=weight_matrix,n_dimensions=2)
    
    print(f"Stress (Non-metric MDS): {stress:.4f}")
    
    # Plot the embedding (with connectivity stats and drawn lines if desired)
    title = f"Non-metric MDS (d_method={distance_method})"
    plot_embedding(points, title, keys, distance_matrix=dist_matrix, weight_matrix=weight_matrix, draw_lines=True)
    
    # Add timestamp to filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    points_file_ts = points_file.replace('.npy', f'_{timestamp}.npy')
    keys_file_ts = points_file.replace('.npy', f'_keys_{timestamp}.npy')
    
    # Save both the embedding points and intervention keys with timestamps
    np.save(points_file_ts, points)
    np.save(keys_file_ts, np.array(keys))
    print(f"Embedding points saved to: {points_file_ts}")
    print(f"Intervention keys saved to: {keys_file_ts}")
    
    return points, stress, keys

# -------------------------------
# Additional Utility: Matrix Statistics
# -------------------------------
def matrix_statistics(matrix):
    """
    Computes and prints overall statistics for a given matrix (excluding trivial values).
    """
    matrix = matrix[~np.isnan(matrix)] # Drop NaN values and get valid elements
    non_zero_elements = matrix[matrix != 0]
    non_one_elements = matrix[matrix != 1]
    non_zero_one_elements = matrix[(matrix != 0) & (matrix != 1)]
    overall_max = np.max(non_one_elements) if non_one_elements.size > 0 else 1
    overall_min = np.min(non_zero_elements) if non_zero_elements.size > 0 else 0
    overall_mean = np.mean(matrix)
    overall_median = np.median(matrix)
    overall_var = np.var(matrix)
    
    non_zero_one_mean = np.mean(non_zero_one_elements) if non_zero_one_elements.size > 0 else 0
    non_zero_one_median = np.median(non_zero_one_elements) if non_zero_one_elements.size > 0 else 0
    non_zero_one_var = np.var(non_zero_one_elements) if non_zero_one_elements.size > 0 else 0
    
    if non_zero_one_elements.size > 0:
        diff_from_median = np.abs(non_zero_one_elements - non_zero_one_median)
        values_far_from_median = diff_from_median > 0.1
        percent_far_from_median = (np.sum(values_far_from_median) / non_zero_one_elements.size) * 100
        perc = [1,2,3,4,5,6,7,8,9,10, 20, 30, 40, 50, 60, 70, 80, 90,91,92,93,94,95,96,97,98,99]
        percentiles = np.percentile(non_zero_one_elements, perc)
    else:
        percent_far_from_median = 0
        percentiles = np.zeros(9)
    
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
    print("\nPercentiles:")
    for a,b in zip(perc, percentiles):
        print(f"  {a}th:                 {b}")
    print("\n\n")

# -------------------------------
# Main Execution Example
# -------------------------------
if __name__ == "__main__":
    json_path = 'Decision Matrix\\Difficult Instances\\X_12.json'
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = load_instance_from_json(data)
    
    # Define the distance transformation methods to test
    distance_methods = [
        "linear", 
        # "sqrt", 
        # "arccos",
        # "logistic",
        # "exponential", 
        # "power2",
        # "power1/3",
        # "arctan",
        # "sine"
    ]
    
    # Run the testing function using non-metric MDS (scikit-learn)
    results = test_embeddings(instance, distance_methods, plot=True, top_n=5, mat_stats=True, draw_lines=False)

    # Uncomment below to compute and save a single embedding:
    # points_file = "points.npy"  # Define the file path to save the embedding points
    # distance_method = distance_methods[0]  # Use the first (and only) distance method
    # points, stress, keys = compute_and_save_embedding(instance, distance_method, points_file, mat_stats=True)
    # print("Computed embedding points and stress saved.")
