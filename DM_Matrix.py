from my_data_structs import *
from clustering_regions import get_distance_matrix, classify_interventions_by_park
import json




def main():
    # Load instance
    instance_path = r'Decision Matrix\Problem setups\C_01.json'
    with open(instance_path, "r") as f:
        instance_data = json.load(f)
    instance = load_instance_from_json(instance_data)

    # Load solutions
    solutions_paths = [
        r'Decision Matrix\Alternatives\1\solution_C_01_900.txt',
        r'Decision Matrix\Alternatives\2\C_01_15min.txt'
    ]
    solutions = [Solution(sol_path) for sol_path in solutions_paths]

    # Computations for the instance
    dist_matrix_df = get_distance_matrix("points.npy", "points_keys.npy") #df with values close, mid, far
    envirnomental_risk_groups = classify_interventions_by_park("points.npy", "points_keys.npy", near_distance=0.05) #dict of keys: high, mid, low

    # Computations for each solution
    for sol in solutions:
        sol.compute_concurrency(instance) #attr1 (max concurrency?)
        sol.compute_seansonality(instance) #attr2 (winter-like, summer-like, is-like proportions)
        sol.plot_concurrency()

        sol.set_worst_risks(instance) #attr3 (highest risk)

        sol.compute_risk_concurrency(instance) #attr4 (quadratic/linear score)
        sol.compute_size_concurrency(instance) #attr5 (quadratic/linear score)
        sol.dist_matrix_to_closeness_concurrency(dist_matrix_df) #attr6 (linear score due to transitivity)
        sol.compute_environmental_impact_concurrency(envirnomental_risk_groups) #attr7 (quadratic/linear score)

        sol.plot_all_concurrency_details()

        
if __name__ == "__main__":
    main()



    
    