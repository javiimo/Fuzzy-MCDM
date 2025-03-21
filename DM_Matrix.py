from my_data_structs import *
from clustering_regions import get_distance_matrix, classify_interventions_by_park
import json
import pandas as pd
import matplotlib.pyplot as plt

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
    solutions = [Solution(sol_path) for sol_path in solutions_paths]

    # Instance-level computations.
    dist_matrix_df = get_distance_matrix(points, point_keys)  # DataFrame with values: "close", "mid", "far"
    envirnomental_risk_groups = classify_interventions_by_park(points, point_keys, near_distance=0.05)  # dict with keys: high, mid, low

    # Compute all metrics for each solution.
    for sol in solutions:
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



def main():
    instance_path = r'Decision Matrix\Problem setups\C_10.json'
    solutions_paths = [
        r'Decision Matrix\Alternatives\1\solution_C_10_900.txt',
        r'Decision Matrix\Alternatives\2\C_10_15min.txt'
    ]
    points = "points_20250310_123528.npy"
    point_keys = "points_keys_20250310_123528.npy"

    DM_matrix = build_DM_matrix(instance_path, solutions_paths, points, point_keys, plots = False)
    print(f"\nDECISION MAKING MATRIX:\n")
    # Create a figure and axis with larger size for better readability
    plt.figure(figsize=(12, 6))
    
    # Format column labels to be multiline
    col_labels = ['\n'.join(col.split()) for col in DM_matrix.columns]
    
    # Plot the table
    table = plt.table(cellText=DM_matrix.values.round(3),
                     colLabels=col_labels,
                     rowLabels=DM_matrix.index,
                     cellLoc='center',
                     loc='center')
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Make column header cells taller to accommodate multiline text
    for cell in table._cells:
        if cell[0] == 0:  # Header row
            table._cells[cell].set_height(0.15)
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title('Decision Making Matrix', pad=20, size=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig('decision_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

        
if __name__ == "__main__":
    main()



    
    