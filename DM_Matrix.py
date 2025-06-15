from my_data_structs import *
from map import build_fuzzy_intervention_distance, build_fuzzy_intervention_park_distance, load_national_parks
from fuzzy_var import fuzz_dist
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import numpy as np
import geopandas as gpd # Assuming parks_gdf is a GeoDataFrame


def compute_intervention_difference_matrix(solutions: List[Solution], plot: bool = False) -> np.ndarray:
    """
    Computes a square matrix where each element [i, j] is the count of interventions 
    that have a different start time between solutions[i] and solutions[j].
    
    Parameters:
        solutions (List[Solution]): List of Solution objects.
        plot (bool): If True, plots a heatmap of the matrix.
        
    Returns:
        np.ndarray: The computed square matrix.
    """
    n = len(solutions)
    matrix = np.zeros((n, n), dtype=int)
    
    # Compute the difference count for each pair of solutions.
    for i in range(n):
        for j in range(i, n):
            # Get the union of intervention keys from both solutions.
            keys = set(solutions[i].intervention_starts.keys()).union(solutions[j].intervention_starts.keys())
            # Count the number of keys with differing start times.
            diff_count = sum(
                1 for key in keys 
                if solutions[i].intervention_starts.get(key) != solutions[j].intervention_starts.get(key)
            )
            matrix[i, j] = diff_count
            matrix[j, i] = diff_count  # The matrix is symmetric.
    
    # Plot a heatmap if requested.
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.title("Intervention Start Time Differences")
        plt.xlabel("Solution Index")
        plt.ylabel("Solution Index")
        plt.colorbar(label="Difference Count")
        plt.show()
    
    return matrix

def build_DM_matrix(instance_path, solutions_paths, points="points.npy", point_keys="points_keys.npy", parks_path="parks.csv", plots=True):
    """
    Build the Decision Matrix (DM Matrix) for evaluating maintenance scheduling alternatives.
    
    This function performs the following steps:
      1. Loads a maintenance scheduling instance from a JSON file.
      2. Loads a list of solution alternatives from the provided file paths.
      3. For each solution, computes several performance criteria, expanding fuzzy concurrency metrics into their classes.
      4. Aggregates these criteria into a pandas DataFrame (the DM Matrix).
    
    Args:
        instance_path (str): Path to the JSON file containing the instance data.
        solutions_paths (List[str]): List of file paths for the solution files.
        points (str): Path to the numpy file for intervention coordinates.
        point_keys (str): Path to the numpy file for intervention keys.
        parks_path (str): Path to the csv for parks data.
        plots (bool, optional): If True, generates plots for each solution. Defaults to True.
    
    Returns:
        pd.DataFrame: A DataFrame representing the DM Matrix.
    """
    # Load instance data and create an instance object.
    with open(instance_path, "r") as f:
        instance_data = json.load(f)
    instance = load_instance_from_json(instance_data)

    # Load all solutions from provided file paths.
    print("Loading solution files...")
    solutions = [Solution(sol_path) for sol_path in solutions_paths]

    # Instance-level computations for fuzzy spatial metrics.
    print("Instance level computations for fuzzy spatial metrics...")
    pts = np.load(points)
    pkeys = np.load(point_keys, allow_pickle=True)
    
    # Define fuzzy distances as specified
    fuzzy_interv = fuzz_dist(5_000, 15_000, 50_000, 100_000, 200_000)
    fuzzy_parks = fuzz_dist(1_000, 10_000, 20_000, 60_000, 150_000)
    
    # Load parks data
    try:
        parks_gdf = load_national_parks(parks_path)
    except Exception as e:
        print(f"Error loading parks shapefile from {parks_path}: {e}")
        print("Cannot compute environmental impact concurrency. Exiting.")
        return pd.DataFrame()

    interv_mems = build_fuzzy_intervention_distance(pts, pkeys, fuzzy_interv)
    park_mems = build_fuzzy_intervention_park_distance(pts, pkeys, parks_gdf, fuzzy_parks)


    # Compute all metrics for each solution.
    for sol in tqdm(solutions, desc="Computing metrics for solutions"):
        sol.compute_concurrency(instance)
        sol.compute_seansonality(instance)
        if plots:
            sol.plot_concurrency()

        sol.set_worst_risks(instance)  # Still needed for some internal logic, but won't add it to the DM matrix
        sol.compute_risk_concurrency(instance)
        sol.compute_size_concurrency(instance)
        sol.dist_matrix_to_closeness_concurrency(interv_mems)
        sol.compute_environmental_impact_concurrency(park_mems, tconorm=np.maximum)

        if plots:
            sol.plot_all_concurrency_details()

    print("Plotting the differences between solutions in a heatmap...")
    mat = compute_intervention_difference_matrix(solutions, plot = plots)

    # Build the DM Matrix as a list of dictionaries.
    print("Building DM Matrix...")
    alternatives = []

    def _add_scores(data_dict, scores_dict, prefix):
        if scores_dict:
            for k, v in scores_dict.items():
                # Sanitize key for column name if needed, though current keys are fine
                col_name = f"{prefix}_{k.replace('-', '_')}" 
                data_dict[col_name] = v

    for i, sol in enumerate(solutions):
        alt_id = f"A{i+1}"
        row_data = {
            "Alternative": alt_id,
            "Highest Concurrency": getattr(sol, "highest_concurrency", None),
        }
        
        # Expand all concurrency scores from dictionaries into separate columns
        _add_scores(row_data, getattr(sol, "size_concurrency", {}), "Size")
        _add_scores(row_data, getattr(sol, "risk_concurrency", {}), "Risk")
        _add_scores(row_data, getattr(sol, "closeness_concurrency", {}), "Closeness")
        _add_scores(row_data, getattr(sol, "environmental_impact_concurrency", {}), "EnvImpact")

        # Add seasonality proportions as separate columns.
        seasonality = getattr(sol, "seasonality", {})
        for season, proportion in seasonality.items():
            row_data[f"{season}-like"] = proportion

        alternatives.append(row_data)

    # Create a pandas DataFrame with alternatives as rows.
    DM_matrix = pd.DataFrame(alternatives)
    DM_matrix.set_index("Alternative", inplace=True)

    return DM_matrix

def get_solution_paths(base_path = r'Decision Matrix\Alternatives\X12'):
    """
    Returns a list of relative paths to non-empty solution files in the X12 folder structure.
    Each team has their own subfolder containing solution files.
    
    Returns:
        list: List of relative paths to non-empty solution files
    """
    solution_paths = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                # Check if file is not empty
                if os.path.getsize(full_path) > 0:
                    # Keep the full path from Decision Matrix
                    solution_paths.append(full_path)
    
    return solution_paths

def extract_solution_keys(solution_paths):
    """
    Extracts key information (team number, time, seed) from solution file paths.

    Args:
        solution_paths (list): List of solution file paths
        
    Returns:
        list: List of dictionaries containing team number, time and seed for each solution
    """
    solution_keys = []
    pattern = r'sol(\d+)_t(\d+)_X_12_s(\d+)\.txt'
    for path in solution_paths:
        filename = os.path.basename(path)
        match = re.match(pattern, filename)
        if match:
            time, team, seed = match.groups()
            solution_keys.append({
                'team': int(team), 'time': int(time), 'seed': int(seed), 'path': path
            })
    return solution_keys

def main():
    instance_path = r'Decision Matrix\Difficult Instances\X_12.json'
    print("Loading the solution paths...")
    solutions_paths = get_solution_paths(base_path=r'Decision Matrix\Alternatives\X12')
    solutions_paths = solutions_paths[:-10]  # Remove the duplicated solutions
    solution_keys = extract_solution_keys(solutions_paths)
    
    # Define paths for spatial data
    points = "points_20250329_203043.npy"
    point_keys = "points_keys_20250329_203043.npy"
    parks_csv_path = "geojsons_nat_parks/pnr_polygon.csv" 

    # Build the new Decision Matrix
    DM_matrix = build_DM_matrix(
        instance_path, solutions_paths, 
        points, point_keys, parks_path=parks_csv_path,
        plots=False
    )
    
    if DM_matrix.empty:
        print("DM Matrix could not be generated.")
        return

    print(f"\nSaving DM Matrix to a CSV file...\n")
    DM_matrix.to_csv('decision_matrix_expanded.csv')
    
    print(f"\nSaving DM Matrix to a markdown file...\n")
    markdown_table = "# Decision Making Matrix\n\n"
    header = "| |" + "|".join(DM_matrix.columns) + "|\n"
    separator = "|---|" + "|".join([":---:" for _ in DM_matrix.columns]) + "|\n"
    markdown_table += header + separator
    
    for i, (idx, row) in enumerate(DM_matrix.iterrows()):
        solution = solution_keys[i]
        row_name = f"T{solution['team']}_D{solution['time']}_S{solution['seed']}"
        
        # Format each value to 3 decimal places, handling potential None values
        formatted_row = [f"{val:.3f}" if isinstance(val, (int, float)) else str(val) for val in row]
        row_str = f"|{row_name}|" + "|".join(formatted_row) + "|\n"
        markdown_table += row_str
        
    with open('decision_matrix_expanded.md', 'w') as f:
        f.write(markdown_table)
        
if __name__ == "__main__":
    main()