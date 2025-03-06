from my_data_structs import *
import numpy as np


def compute_risk_corr_np(risk1, risk2, alpha=1e-15):
    """
    Computes a modified correlation between two risk series that captures jump sizes.
    
    For each step:
      - Compute the differences:
            diff1 = risk1[i+1] - risk1[i]
            diff2 = risk2[i+1] - risk2[i]
      - Compute a similarity measure based on the difference of jumps:
            similarity = exp(-alfa|diff1 - diff2|)
      - Determine sign:
            sign = +1 if diff1 * diff2 >= 0, else -1
      - Contribution for this step:
            contribution = sign * similarity
    
    The final correlation is the sum of the contributions.
    
    Parameters:
        risk1 (array-like): First risk series.
        risk2 (array-like): Second risk series.
    
    Returns:
        float: The computed modified correlation.
    """
    # Compute consecutive differences for each risk series
    diff1 = np.diff(risk1)
    diff2 = np.diff(risk2)
    
    # Similarity: the closer diff1 and diff2 are, the closer to 1 the similarity.
    similarity = np.exp(-alpha * np.abs(diff1 - diff2)) # values in (0,1]
    
    # Determine sign: if both jumps have the same sign, contribution is positive.
    sign = np.where(diff1 * diff2 >= 0, 1, -1)
    
    # Compute contributions for each step
    contributions = sign * similarity
    return np.sum(contributions)


def compute_risk_corr_matrix(instance):
    """
    Computes a normalized correlation matrix for the interventions in instance.
    It returns a tuple (keys, norm_corr_matrix) where:
      - keys: list of intervention identifiers.
      - norm_corr_matrix: a matrix with normalized correlations.
    
    The normalized correlation between i and j is defined as:
       norm_corr(i,j) = corr(i,j) / sqrt(corr(i,i)*corr(j,j))
    
    Parameters:
        instance (object): An instance containing interventions with risk time series.
    
    Returns:
        tuple: (list of intervention names, NumPy array of the correlation matrix)
    """
    interventions = instance.interventions
    keys = list(interventions.keys())  # Extract intervention names
    n = len(keys)  # Number of interventions
    corr_matrix = np.zeros((n, n))  # Initialize empty correlation matrix
    
    for i in range(n):
        # Convert the risk time series to a NumPy array
        risk_i = np.array(interventions[keys[i]].overall_mean_risk)
        for j in range(i, n):  # Iterate over upper triangle (including diagonal)
            risk_j = np.array(interventions[keys[j]].overall_mean_risk)
            corr = compute_risk_corr_np(risk_i, risk_j)
            corr_matrix[i, j] = corr  # Fill upper triangle
            corr_matrix[j, i] = corr  # Fill lower triangle (symmetry)

    # Normalize the correlations so that the diagonal is 1
    norm_corr_matrix = np.zeros_like(corr_matrix)
    for i in range(n):
        for j in range(n):
            norm_corr_matrix[i, j] = corr_matrix[i, j] / np.sqrt(corr_matrix[i, i] * corr_matrix[j, j])
    
    return keys, norm_corr_matrix

def compute_distance_matrix(instance):
    """
    Computes a distance matrix from the normalized correlation matrix.
    Uses the transformation:
         distance = sqrt((1 - normalized_correlation))
    to ensure the distances lie in [0, 1].
    
    Parameters:
        instance (object): An instance containing interventions with risk time series.
        alpha (float): Parameter to pass to the risk correlation computation.
    
    Returns:
        tuple: (list of intervention names, NumPy array of the distance matrix)
    """
    keys, norm_corr_matrix = compute_risk_corr_matrix(instance)
    #distance_matrix = np.sqrt((1 - norm_corr_matrix))
    distance_matrix = (1 - norm_corr_matrix)
    
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    d_min = np.min(distance_matrix[mask])
    d_max = np.max(distance_matrix[mask])
    epsilon = 1e-6  # Small value to avoid zero distances
    distance_matrix[mask] =  epsilon + (1 - epsilon) * (distance_matrix[mask] - d_min) / (d_max - d_min)
    return keys, distance_matrix



import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def recover_points_MDS(distance_matrix, n_dimensions=2):
    """
    Recovers the original point coordinates from a distance matrix using Multidimensional Scaling (MDS).
    
    :param distance_matrix: A square NumPy array representing the pairwise distance matrix.
    :param n_dimensions: Number of dimensions to recover (default: 2).
    :return: Recovered coordinates as a NumPy array.
    """
    mds = MDS(n_components=n_dimensions, dissimilarity="precomputed", random_state=42)
    points = mds.fit_transform(distance_matrix)

    #Check how good the fit is
    print("MDS Stress:", mds.stress_)
    from sklearn.metrics import pairwise_distances
    recovered_distance_matrix = pairwise_distances(points)
    difference_matrix = distance_matrix - recovered_distance_matrix
    # Compute the Root Mean Squared Error (RMSE) between the distances.
    rmse = np.sqrt(np.mean((difference_matrix) ** 2))
    print("RMSE of distance differences:", rmse)
    # Compute median distance (excluding diagonal zeros) and compare with RMSE
    mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    median_distance = np.median(distance_matrix[mask])
    print("Median distance in the original distance matrix:", median_distance)
    print("RMSE as percentage of median distance:", (rmse / median_distance) * 100, "%") #acceptable up to a 20% approx
    return points



if __name__ == "__main__":
    json_path = 'Decision Matrix\Problem setups\C_01.json'
    with open(json_path, "r") as f:
        data = json.load(f)
    instance = load_instance_from_json(data)
    keys, distance_matrix = compute_distance_matrix(instance)
    print("Intervention Keys:", keys)
    print("Distance Matrix:\n", distance_matrix)

    ####################################################
    #         Statistics of the distance matrix
    ####################################################
    # Compute overall statistics (exclude zeros for min)
    non_zero_elements = distance_matrix[distance_matrix != 0]
    overall_max = np.max(distance_matrix)
    overall_min = np.min(non_zero_elements) if non_zero_elements.size > 0 else 0
    overall_mean = np.mean(distance_matrix)
    overall_var = np.var(distance_matrix)
    
    print("\nOverall Distance Matrix Stats:")
    print("  Max value:           ", overall_max)
    print("  Min value (non-zero):", overall_min)
    print("  Mean value:          ", overall_mean)
    print("  Variance:            ", overall_var)
    
    # # Compute and print per-row (and column) stats excluding diagonal zeros.
    # print("\nRow/Column Stats (excluding diagonal zeros):")
    # for i, row in enumerate(distance_matrix):
    #     # Exclude the diagonal zero entry by filtering out zeros.
    #     row_non_zero = row[row != 0]
    #     row_min = np.min(row_non_zero) if row_non_zero.size > 0 else 0
    #     row_max = np.max(row_non_zero) if row_non_zero.size > 0 else 0
    #     row_mean = np.mean(row_non_zero) if row_non_zero.size > 0 else 0
    #     row_var = np.var(row_non_zero) if row_non_zero.size > 0 else 0
    #     print(f"  {keys[i]}:")
    #     print(f"    Min (non-zero): {row_min}")
    #     print(f"    Max:            {row_max}")
    #     print(f"    Mean:           {row_mean}")
    #     print(f"    Variance:       {row_var}")


    ####################################################
    #         Creating a map from the distance matrix
    ####################################################


    recovered_points = recover_points_MDS(distance_matrix)

    # Plot results
    plt.scatter(recovered_points[:, 0], recovered_points[:, 1], c='red', marker='o')
    for i, (x, y) in enumerate(recovered_points):
        plt.text(x, y, f"P{i}", fontsize=12, ha='right')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Recovered Points from Distance Matrix")
    plt.grid()
    plt.show()
