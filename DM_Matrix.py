from my_data_structs import *
from clustering_regions import get_distance_matrix, classify_interventions_by_park
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm

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

def build_DM_matrix(instance_path, solutions_paths, points = "points.npy", point_keys = "points_keys.npy", plots=True):
    """
    Build the Decision Matrix (DM Matrix) for evaluating maintenance scheduling alternatives.
    
    This function performs the following steps:
      1. Loads a maintenance scheduling instance from a JSON file.
      2. Loads a list of solution alternatives from the provided file paths.
      3. For each solution, computes several performance criteria:
         - Highest concurrency: The maximum number of interventions active at any timestep.
         - Worst risk: The highest risk value among scheduled interventions.
         - Risk concurrency: A quadratic/linear score based on the clustering of interventions by risk.
         - Size concurrency: A quadratic/linear score based on the clustering of interventions by intervention size.
         - Closeness concurrency: A linear score computed from the number of close and mid-distance intervention pairs. This one is linear due to the transitivity of the distance relation.
         - Environmental impact concurrency: A score based on concurrent high and mid environmental impact interventions.
         - Seasonality proportions: For each season (e.g., winter, summer, etc.), the proportion of scheduled interventions active during that season.
      4. Aggregates these criteria into a pandas DataFrame (the DM Matrix), where each row represents a solution alternative 
         (named “A1”, “A2”, “A3”, etc.) and each column corresponds to one criterion.
    
    Args:
        instance_path (str): Path to the JSON file containing the maintenance scheduling instance data.
        solutions_paths (List[str]): List of file paths for the solution files.
        plots (bool, optional): If True, generates plots for each solution during processing. Defaults to True.
    
    Returns:
        pd.DataFrame: A DataFrame representing the DM Matrix with alternatives as rows and evaluation criteria as columns.
    """
    # Load instance data and create an instance object.
    with open(instance_path, "r") as f:
        instance_data = json.load(f)
    instance = load_instance_from_json(instance_data)

    # Load all solutions from provided file paths.
    print("Loading solution files...")
    solutions = [Solution(sol_path) for sol_path in solutions_paths]

    # Instance-level computations.
    print("Instance level computations...")
    dist_matrix_df = get_distance_matrix(points, point_keys)  # DataFrame with values: "close", "mid", "far"
    envirnomental_risk_groups = classify_interventions_by_park(points, point_keys, near_distance=0.05)  # dict with keys: high, mid, low

    # Compute all metrics for each solution.
    for sol in tqdm(solutions, desc="Computing metrics for solutions"):
        sol.compute_concurrency(instance)           # Computes concurrency (e.g., highest concurrency)
        sol.compute_seansonality(instance)            # Computes seasonality proportions (e.g., winter, summer, etc.)
        if plots:
            sol.plot_concurrency()

        sol.set_worst_risks(instance)                 # Sets worst risk (highest risk among interventions)
        sol.compute_risk_concurrency(instance)        # Computes risk concurrency score
        sol.compute_size_concurrency(instance)        # Computes size concurrency score
        sol.dist_matrix_to_closeness_concurrency(dist_matrix_df)  # Computes closeness concurrency score
        sol.compute_environmental_impact_concurrency(envirnomental_risk_groups)  # Computes environmental impact concurrency score

        if plots:
            sol.plot_all_concurrency_details()

    print("Plotting the differences between solutions in a heatmap...")
    mat = compute_intervention_difference_matrix(solutions, plot = True) #This is the similarity between sols matrix

    # Build the DM Matrix as a list of dictionaries.
    print("Building DM Matrix...")
    alternatives = []
    for i, sol in enumerate(solutions):
        alt_id = f"A{i+1}"
        # Extract key criteria values from the solution.
        row_data = {
            "Alternative": alt_id,
            "Highest Concurrency": getattr(sol, "highest_concurrency", None),
            "Highest Risk": getattr(sol, "highest_risk", None),
            "Risk Concurrency": getattr(sol, "risk_concurrency", None),
            "Size Concurrency": getattr(sol, "size_concurrency", None),
            "Closeness Concurrency": getattr(sol, "closeness_concurrency", None),
            "Environmental Impact Concurrency": getattr(sol, "env_impact_concurrency", None)
        }
        # Add seasonality proportions as separate columns.
        # For example, if sol.seasonality contains keys like 'winter', 'summer', 'is'
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
    
    for path in solution_paths:
        # Get just the filename from the path
        filename = os.path.basename(path)
        
        # Extract components using regex pattern matching
        pattern = r'sol(\d+)_t(\d+)_X_12_s(\d+)\.txt'
        match = re.match(pattern, filename)
        
        if match:
            time, team, seed = match.groups()
            solution_keys.append({
                'team': int(team),
                'time': int(time),
                'seed': int(seed),
                'path': path
            })
            
    return solution_keys



def main():
    instance_path = r'Decision Matrix\Difficult Instances\X_12.json'
    print("Loading the solution paths...")
    solutions_paths = get_solution_paths(base_path = r'Decision Matrix\Alternatives\X12')
    solutions_paths = solutions_paths[:-10] #Remove the duplicated solutions
    solution_keys = extract_solution_keys(solutions_paths)
    points = "points_20250329_203043.npy"
    point_keys = "points_keys_20250329_203043.npy"
    DM_matrix = build_DM_matrix(instance_path, solutions_paths, points, point_keys, plots = True)

    print(f"\nSaving DM Matrix to a CSV file...\n")
    DM_matrix.to_csv('decision_matrix.csv')
    
    print(f"\nSaving DM Matrix to a markdown file...\n")
    # Save table as markdown
    markdown_table = "# Decision Making Matrix\n\n"
    
    # Add header row
    header = "| |" + "|".join(DM_matrix.columns) + "|\n"
    separator = "|---|" + "|".join([":---:" for _ in DM_matrix.columns]) + "|\n"
    markdown_table += header + separator
    
    # Add data rows
    for i, (idx, row) in enumerate(DM_matrix.iterrows()):
        # Get solution info from solution_keys
        solution = solution_keys[i]
        row_name = f"T{solution['team']}_D{solution['time']}_S{solution['seed']}"
        row_str = f"|{row_name}|" + "|".join([f"{val:.3f}" for val in row]) + "|\n"
        markdown_table += row_str
        
    # Save to file
    with open('decision_matrix.md', 'w') as f:
        f.write(markdown_table)

        
if __name__ == "__main__":
    main()



    
    