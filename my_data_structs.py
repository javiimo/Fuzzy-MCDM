import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


####################################################
# Auxiliary functions
####################################################

# Set up the logger for errors and warnings
logger = logging.getLogger("error_logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("error.log")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def compute_stats(data: List[float]) -> str:
    """Compute basic statistics (min, max, mean, stdev) for a list of numbers."""
    if not data:
        return "no data"
    try:
        avg = statistics.mean(data)
        deviation = statistics.stdev(data) if len(data) > 1 else 0
        return f"min={min(data)}, max={max(data)}, mean={avg:.2f}, stdev={deviation:.2f}"
    except Exception as e:
        return f"error: {e}"

def trim_list(lst: List[Any]) -> str:
    """
    Return a string representation of a list. If the list has more than 20 elements,
    only the first 20 are shown followed by an ellipsis.
    """
    if not isinstance(lst, list):
        return str(lst)
    if len(lst) > 20:
        return str(lst[:20]) + " ... (total: " + str(len(lst)) + " elements)"
    return str(lst)


######################################################
#               Clustering Functions
######################################################

def cluster_by_attribute(values: List[float], names: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """
    Generic clustering function that applies k-means (k=3) to 1D values,
    and maps the clusters (in ascending order of centroid value) to the provided labels.
    
    Additionally, prints the centroid value for each cluster label.
    
    Args:
        values: A list of one-dimensional real values.
        names: A list of instance names corresponding to the values.
        labels: A list of 3 cluster labels ordered from smallest to largest.
                e.g. for sizes: ['small', 'mid', 'big'], for risk: ['low', 'mid', 'high'].
                
    Returns:
        A dictionary mapping each label to the list of instance names assigned to that cluster.
    """
    data = np.array(values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_ids = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_.flatten()
    
    # Determine mapping: smallest centroid gets labels[0], etc.
    sorted_cluster_ids = np.argsort(centroids)
    cluster_label_mapping = {cluster_id: labels[idx] for idx, cluster_id in enumerate(sorted_cluster_ids)}
    
    # Informative print of the centroids for each label.
    print("Cluster centroids:")
    for idx, label in enumerate(labels):
        centroid_value = centroids[sorted_cluster_ids[idx]]
        print(f"  {label}: {centroid_value:.4f}")
    
    # Build the result dictionary.
    result: Dict[str, List[str]] = {label: [] for label in labels}
    for name, cluster_id in zip(names, cluster_ids):
        result[cluster_label_mapping[cluster_id]].append(name)
        
    return result

def plot_clustering_heatmap(cluster_dict: Dict[str, List[str]]) -> None:
    """
    Plots a heatmap for the clustering result contained in cluster_dict.
    The input dictionary has cluster labels as keys and lists of instance names as values.
    The function creates a one-row heatmap where each cell's color corresponds to the cluster assignment.
    
    Args:
        cluster_dict: A dictionary mapping cluster labels (e.g., 'small', 'mid', 'big')
                      to a list of instance names in that cluster.
    """
    # Determine the order of clusters. Assuming the keys are in the desired order,
    # e.g., for sizes: ['small', 'mid', 'big'] or for risk: ['low', 'mid', 'high'].
    cluster_order = list(cluster_dict.keys())
    
    # Create lists for cluster assignments and corresponding intervention names.
    assignments = []
    names = []
    for idx, cluster_label in enumerate(cluster_order):
        for name in cluster_dict[cluster_label]:
            assignments.append(idx)
            names.append(name)
            
    # Create a 2D array for heatmap: one row with a cell per intervention.
    data = np.array(assignments).reshape(1, -1)
    
    # Create the heatmap.
    fig, ax = plt.subplots(figsize=(max(8, len(names)*0.5), 2))
    cax = ax.imshow(data, aspect="auto", cmap="viridis")
    
    # Set ticks on x-axis for each intervention.
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    # One y-tick for the single row.
    ax.set_yticks([0])
    ax.set_yticklabels([""])
    
    # Create a colorbar with ticks corresponding to the cluster indices.
    cbar = fig.colorbar(cax, ax=ax, ticks=range(len(cluster_order)))
    cbar.ax.set_yticklabels(cluster_order)
    
    ax.set_title("Interventions Clustering Heatmap")
    plt.tight_layout()
    plt.show()

def cluster_interventions_by_size(instance) -> Dict[str, List[str]]:
    """
    Clusters the interventions in the maintenance scheduling instance based on mean intervention size.
    Uses labels 'small', 'mid', 'big' for clusters.
    
    Args:
        instance: A MaintenanceSchedulingInstance with interventions.
    
    Returns:
        A dictionary mapping the cluster label to the list of intervention names.
    """
    names = []
    sizes = []
    for intervention in instance.interventions.values():
        # Ensure the mean size is computed.
        size = intervention.mean_intervention_size
        if size == 0.0:
            size = intervention.compute_mean_intervention_size()
        names.append(intervention.name)
        sizes.append(size)
        
    # For sizes: smallest value -> "small", largest -> "big".
    print("Cluster by size centroids:")
    return cluster_by_attribute(sizes, names, ['small', 'mid', 'big'])

def cluster_interventions_by_risk(instance) -> Dict[str, List[str]]:
    """
    Clusters the interventions in the maintenance scheduling instance based on average overall risk.
    Uses labels 'low', 'mid', 'high' for clusters.
    
    Args:
        instance: A MaintenanceSchedulingInstance with interventions.
    
    Returns:
        A dictionary mapping the cluster label to the list of intervention names.
    """
    names = []
    avg_risks = []
    for intervention in instance.interventions.values():
        # Compute overall mean risk if needed.
        if not intervention.overall_mean_risk:
            intervention.compute_overall_mean_risk()
        # Calculate average risk (avoiding division by zero).
        if intervention.overall_mean_risk:
            avg_risk = sum(intervention.overall_mean_risk) / len(intervention.overall_mean_risk)
        else:
            avg_risk = 0.0
        names.append(intervention.name)
        avg_risks.append(avg_risk)
    
    # For risk: lowest value -> "low", highest -> "high".
    print("Cluster by risk centroids:")
    return cluster_by_attribute(avg_risks, names, ['low', 'mid', 'high'])

######################################################
#           Data Classes
######################################################


@dataclass
class Resource:
    """
    Represents a resource available for maintenance.
    """
    name: str
    capacity_min: List[float]  # One value per period (length T)
    capacity_max: List[float]  # One value per period (length T)

    def __str__(self) -> str:
        stats_min = compute_stats(self.capacity_min)
        stats_max = compute_stats(self.capacity_max)
        return (f"Resource '{self.name}':\n"
                f"  Periods (T): {len(self.capacity_min)}\n"
                f"  Capacity min: {self.capacity_min} ({stats_min})\n"
                f"  Capacity max: {self.capacity_max} ({stats_max})")

@dataclass
class Season:
    """
    Represents a season grouping certain time periods.
    """
    name: str
    periods: List[int]  # Time periods belonging to this season

    def __str__(self) -> str:
        if self.periods:
            return (f"Season '{self.name}': {len(self.periods)} period(s), "
                    f"range: {min(self.periods)}-{max(self.periods)}, periods: {self.periods}")
        else:
            return f"Season '{self.name}': no periods assigned."

@dataclass
class Exclusion:
    """
    Represents a mutual exclusion constraint: the given interventions
    must not overlap during the specified season.
    """
    interventions: List[str]
    season: str

    def __str__(self) -> str:
        return (f"Exclusion Constraint: Interventions {self.interventions} "
                f"cannot overlap during season '{self.season}'.")


@dataclass
class ScheduleOption:
    """
    Represents one scheduling option for an intervention when started at a specific time.
    
    Attributes:
        start_time: The potential start time (e.g., day or period number).
        duration: The length of the intervention when started at this time.
        workloads: A mapping from resource name to a list of workload values
                   over the duration of the intervention. For example, if resource "c1"
                   is used only in the first period, it might be: {"c1": [14, 0, 0]}.
        risks: A list (over the intervention's timesteps) where each element is a 
               list of risk values—one per scenario—for that period.
               For example: [[4,8,2], [0,0], [0,0,0]].
        mean_risk: A list of mean risk values calculated for each timestep for THIS start time across all scenarios. 
                   Only using non-zero risks.
        worst_risk: The highest risk value across all timesteps and scenarios.
        option_size: The sum of all workload values across all resources and timesteps and multiply by the duration
    """
    start_time: int
    duration: int
    workloads: Dict[str, List[float]]
    risks: List[List[float]]
    mean_risk: List[float] = field(default_factory=list)
    worst_risk: float = field(default=0.0)  # Highest risk across timesteps and scenarios
    option_size: float = field(default=0.0)  # New attribute to store total workload

    def __post_init__(self):
        if self.risks:
            self.mean_risk = []
            for timestep_risks in self.risks:
                # Only consider non-zero risk values.
                non_zero_risks = [value for value in timestep_risks if value > 0]
                total_risk = sum(non_zero_risks)
                total_scenarios = len(non_zero_risks)
                mean = total_risk / total_scenarios if total_scenarios > 0 else 0.0
                self.mean_risk.append(mean)
            # Compute worst_risk as the maximum risk value across all timesteps and scenarios.
            all_risks = [value for timestep in self.risks for value in timestep]
            self.worst_risk = max(all_risks) if all_risks else 0.0
        else:
            self.mean_risk = []
            self.worst_risk = 0.0
        
        # Compute option_size by summing all workload values in all lists.
        self.option_size = sum(sum(workload) for workload in self.workloads.values())*self.duration if self.workloads else 0.0

    def __str__(self) -> str:
        # Format workloads using compute_stats (assumed to be defined elsewhere).
        if self.workloads:
            workloads_str = "\n    ".join(
                f"{res}: {vals} ({compute_stats(vals)})"
                for res, vals in self.workloads.items()
            )
        else:
            workloads_str = "None"
        # Format the risk profiles per timestep.
        if self.risks:
            risks_str = "\n    ".join(
                f"Relative period {i+1}: {r} ({compute_stats(r)})"
                for i, r in enumerate(self.risks)
            )
        else:
            risks_str = "None"
        # Properly format the mean_risk list.
        mean_risk_str = (
            "[" + ", ".join(f"{mr:.2f}" for mr in self.mean_risk) + "]"
            if self.mean_risk else "[]"
        )
        return (f"Schedule Option (start_time={self.start_time}, duration={self.duration}, mean_risk={mean_risk_str}, option_size={self.option_size}):\n"
                f"  Workloads:\n    {workloads_str}\n"
                f"  Risks:\n    {risks_str}")

@dataclass
class Intervention:
    """
    Represents an intervention (maintenance task) to be scheduled.
    
    Attributes:
        name: Unique identifier for the intervention.
        tmax: The last period at which the intervention may start so that it finishes on time.
        options: A list of possible scheduling options for the intervention.
        overall_mean_risk: A list of mean risk values calculated for each timestep across all options (i.e. across all start times).
    """
    name: str
    tmax: int
    options: List[ScheduleOption] = field(default_factory=list)
    overall_mean_risk: List[float] = field(init=False, default_factory=list)
    mean_intervention_size: float = field(init=False, default=0.0)
    
    def add_option(self, option: ScheduleOption):
        if option.start_time > self.tmax:
            raise ValueError(
                f"Option start time {option.start_time} exceeds tmax ({self.tmax}) for intervention {self.name}"
            )
        self.options.append(option)
        # Update the overall_mean_risk after adding a new option.
        #self.compute_overall_mean_risk() #Not efficient, compute it once everything is initialized.

    def compute_overall_mean_risk(self) -> List[float]:
        """
        Computes the overall mean risk per timestep across all schedule options,
        using only the mean risk values that are greater than 0.
        """
        if not self.options:
            self.overall_mean_risk = []
            return self.overall_mean_risk
        
        # Determine the maximum number of timesteps among all schedule options.
        max_timesteps = max(len(opt.mean_risk) for opt in self.options)
        overall_means = []
        for t in range(max_timesteps):
            # Gather all positive mean risk values from the t-th timestep of each option.
            timestep_values = []
            for opt in self.options:
                if t < len(opt.mean_risk) and opt.mean_risk[t] > 0:
                    timestep_values.append(opt.mean_risk[t])
            # Compute average if there are any positive values; otherwise, default to 0.
            if timestep_values:
                overall_means.append(sum(timestep_values) / len(timestep_values))
            else:
                overall_means.append(0.0)
        self.overall_mean_risk = overall_means
        return self.overall_mean_risk
    
    def compute_mean_intervention_size(self) -> float:
        """
        Computes the mean intervention size across all schedule options,
        averaging only those options where the option_size is not 0.
        """
        # Gather sizes from options with non-zero option_size.
        sizes = [option.option_size for option in self.options if option.option_size > 0]
        self.mean_intervention_size = sum(sizes) / len(sizes) if sizes else 0.0
        return self.mean_intervention_size

    def __str__(self) -> str:
        # Compute overall_mean_risk if it hasn't been computed yet.
        if not self.overall_mean_risk and self.options:
            self.compute_overall_mean_risk()
        
        overall_risk_str = (
            "[" + ", ".join(f"{risk:.2f}" for risk in self.overall_mean_risk) + "]"
            if self.overall_mean_risk else "Not computed"
        )
        
        options_str = "\n  ".join(str(opt) for opt in self.options) if self.options else "None"
        
        return (
            f"Intervention '{self.name}':\n"
            f"  tmax = {self.tmax}\n"
            f"  Options Count = {len(self.options)}\n"
            f"  Overall Mean Risk = {overall_risk_str}\n"
            f"  Schedule Options:\n  {options_str}"
        )



@dataclass
class MaintenanceSchedulingInstance:
    """
    Represents a complete instance of the maintenance scheduling problem.
    
    Attributes:
        T: Total number of time periods in the planning horizon.
        scenarios_number: A list specifying the number of risk scenarios for each time period.
        resources: Dictionary of Resource objects keyed by resource name.
        seasons: Dictionary of Season objects keyed by season name.
        interventions: Dictionary of Intervention objects keyed by intervention name.
        exclusions: List of mutual exclusion constraints.
    """
    T: int
    scenarios_number: List[int]
    resources: Dict[str, Resource] = field(default_factory=dict)
    seasons: Dict[str, Season] = field(default_factory=dict)
    interventions: Dict[str, Intervention] = field(default_factory=dict)
    exclusions: List[Exclusion] = field(default_factory=list)
    
    def add_resource(self, resource: Resource):
        self.resources[resource.name] = resource
        
    def add_season(self, season: Season):
        self.seasons[season.name] = season
        
    def add_intervention(self, intervention: Intervention):
        self.interventions[intervention.name] = intervention
        
    def add_exclusion(self, exclusion: Exclusion):
        self.exclusions.append(exclusion)

    def __str__(self) -> str:
        resources_str = "\n".join(str(r) for r in self.resources.values())
        seasons_str = "\n".join(str(s) for s in self.seasons.values())
        interventions_str = "\n".join(str(i) for i in self.interventions.values())
        exclusions_str = "\n".join(str(e) for e in self.exclusions)
        return (f"Maintenance Scheduling Instance:\n"
                f"  Planning Horizon (T): {self.T}\n"
                f"  Scenarios per period: {self.scenarios_number}\n"
                f"  Resources:\n{resources_str}\n"
                f"  Seasons:\n{seasons_str}\n"
                f"  Interventions:\n{interventions_str}\n"
                f"  Exclusions:\n{exclusions_str}")

    def show(self, indent: int = 0) -> None:
        """
        Print a detailed, hierarchical view of the instance.
        If any list has more than 20 elements, only the first 20 are shown followed by an ellipsis.
        """
        pad = " " * indent
        print(f"{pad}Maintenance Scheduling Instance:")
        print(f"{pad}  Planning Horizon (T): {self.T}")
        print(f"{pad}  Scenarios per period: {trim_list(self.scenarios_number)}")
        
        # Resources
        print(f"{pad}  Resources:")
        if self.resources:
            for res in self.resources.values():
                print(f"{pad}    Resource: {res.name}")
                print(f"{pad}      Capacity min: {trim_list(res.capacity_min)}")
                print(f"{pad}      Capacity max: {trim_list(res.capacity_max)}")
        else:
            print(f"{pad}    None")
        
        # Seasons
        print(f"{pad}  Seasons:")
        if self.seasons:
            for season in self.seasons.values():
                print(f"{pad}    Season: {season.name}")
                if season.periods:
                    print(f"{pad}      Periods: {trim_list(season.periods)} (range: {min(season.periods)}-{max(season.periods)})")
                else:
                    print(f"{pad}      No periods assigned")
        else:
            print(f"{pad}    None")
        
        # Interventions
        print(f"{pad}  Interventions:")
        if self.interventions:
            for intv in self.interventions.values():
                print(f"{pad}    Intervention: {intv.name}")
                print(f"{pad}      tmax: {intv.tmax}")
                if intv.options:
                    print(f"{pad}      Options:")
                    for option in intv.options:
                        print(f"{pad}        Schedule Option:")
                        print(f"{pad}          Start time: {option.start_time}")
                        print(f"{pad}          Duration: {option.duration}")
                        # Workloads
                        print(f"{pad}          Workloads:")
                        if option.workloads:
                            for res_name, wl in option.workloads.items():
                                print(f"{pad}            Resource {res_name}: {trim_list(wl)}")
                        else:
                            print(f"{pad}            None")
                        # Risks
                        print(f"{pad}          Risks:")
                        if option.risks:
                            for i, risk_list in enumerate(option.risks):
                                print(f"{pad}            Timestep {i+1}: {trim_list(risk_list)}")
                        else:
                            print(f"{pad}            None")
                else:
                    print(f"{pad}      No scheduling options available")
        else:
            print(f"{pad}    None")
        
        # Exclusions
        print(f"{pad}  Exclusions:")
        if self.exclusions:
            for excl in self.exclusions:
                print(f"{pad}    Exclusion Constraint:")
                print(f"{pad}      Interventions: {trim_list(excl.interventions)}")
                print(f"{pad}      Season: {excl.season}")
        else:
            print(f"{pad}    None")



############################################
#               ESQUEMA
############################################
# Original JSON                           New Python Classes
# ---------------------------------------------------------------
# "Resources": {                           ⟶  Resource(name=..., 
#     "c1": {                                    capacity_min=..., 
#        "min": [...],                           capacity_max=...)
#        "max": [...] }
# }                                       

# "Seasons": {                             ⟶  Season(name="winter", periods=[...])
#     "winter": [...],                           Season(name="summer", periods=[...])
#     "summer": [...],                           Season(name="is", periods=[...])
#     "is": [] }
                                       
# "Interventions": {                       ⟶  Intervention(name="I1", tmax=..., 
#     "I1": {                                      options=[
#          "tmax": ... ,                              ScheduleOption(start_time=1, duration=...,
#          "Delta": [...],                              workloads={...}, risks=[...]),
#          "workload": {                                ScheduleOption(start_time=2, ...),
#              "c1": {                                    ...
#                   "1": { ... }, 
#                   "2": { ... },
#                   "3": { ... }
#              }
#          },
#          "risk": {                                   ]
#              "1": { ... },
#              "2": { ... },
#              "3": { ... }
#          }
#     }
# }
                                       
# "Exclusions": {                          ⟶  Exclusion(interventions=[...], season="...")
#     "E1": ["I2", "I3", "full"]
# }

# "T": 3,                                  ⟶  MaintenanceSchedulingInstance.T = 3
# "Scenarios_number": [3,2,3]               ⟶  MaintenanceSchedulingInstance.scenarios_number = [3,2,3]




# ------------------------------------------------------------------------------
# Function to load from the original JSON structure into the new object model
# ------------------------------------------------------------------------------

def load_instance_from_json(json_data: Dict[str, Any]) -> MaintenanceSchedulingInstance:
    """
    Load a MaintenanceSchedulingInstance from a JSON-like dictionary in the original structure.
    
    Mapping Overview:
      "T"                ⟶ MaintenanceSchedulingInstance.T
      "Scenarios_number" ⟶ MaintenanceSchedulingInstance.scenarios_number
      "Resources"        ⟶ Resources -> Resource objects
      "Seasons"          ⟶ Seasons -> Season objects
      "Interventions"    ⟶ Intervention objects with ScheduleOption(s)
         • "tmax"       ⟶ Intervention.tmax
         • "Delta"      ⟶ For each option, ScheduleOption.duration (with start_time given by the option index, 1-indexed)
         • "workload"   ⟶ ScheduleOption.workloads: For each resource, create a list of length T. For each absolute timestep (1 to T),
                              if the JSON specifies a workload for that timestep for the current option's start time, use it;
                              otherwise, leave it as zero.
         • "risk"       ⟶ ScheduleOption.risks: For each absolute timestep (1 to T), if the JSON specifies risks for that timestep
                              for the current option's start time, use that list; otherwise, use an empty list.
      "Exclusions"       ⟶ Exclusion objects (last element is the season, preceding elements are interventions)
    """
    try:
        T = json_data["T"]
        scenarios_number = json_data["Scenarios_number"]
    except Exception as e:
        logger.error(f"Error reading instance horizon or scenarios: {e}", exc_info=True)
        T = 0
        scenarios_number = []
    instance = MaintenanceSchedulingInstance(T=T, scenarios_number=scenarios_number)
    print("Starting to load instance from JSON with T =", T, "and scenarios_number =", scenarios_number)
    
    # Load Resources.
    print("Loading Resources...")
    _total_resources = len(json_data.get("Resources", {}))
    _res_counter = 0
    for res_name, res_data in json_data.get("Resources", {}).items():
        _res_counter += 1
        print(f"Loading resource {_res_counter} of {_total_resources}: {res_name}")
        try:
            resource = Resource(
                name=res_name,
                capacity_min=res_data["min"],
                capacity_max=res_data["max"]
            )
            instance.add_resource(resource)
        except Exception as e:
            logger.error(f"Error loading resource '{res_name}': {e}", exc_info=True)
            continue
    print("Finished loading Resources.")
    
    # Load Seasons.
    print("Loading Seasons...")
    _total_seasons = len(json_data.get("Seasons", {}))
    _season_counter = 0
    for season_name, periods in json_data.get("Seasons", {}).items():
        _season_counter += 1
        print(f"Loading season {_season_counter} of {_total_seasons}: {season_name}")
        try:
            season = Season(name=season_name, periods=periods)
            instance.add_season(season)
        except Exception as e:
            logger.error(f"Error loading season '{season_name}': {e}", exc_info=True)
            continue
    print("Finished loading Seasons.")
    
    # Load Interventions.
    print("Loading Interventions...")
    _total_interventions = len(json_data.get("Interventions", {}))
    _intv_counter = 0
    for intv_name, intv_data in json_data.get("Interventions", {}).items():
        _intv_counter += 1
        intv_name = f"I{intv_name.split('_')[1]}" 
        print(f"Loading intervention {_intv_counter} of {_total_interventions}: {intv_name}")
        try:
            tmax = int(intv_data["tmax"])
        except Exception as e:
            logger.error(f"Error loading tmax for intervention '{intv_name}': {e}", exc_info=True)
            continue
        
        try:
            intervention = Intervention(name=intv_name, tmax=tmax)
        except Exception as e:
            logger.error(f"Error creating intervention '{intv_name}': {e}", exc_info=True)
            continue
        
        delta_list = intv_data.get("Delta", [])
        workload_data = intv_data.get("workload", {})
        risk_data = intv_data.get("risk", {})
        
        _total_options = len(delta_list)
        _option_counter = 0
        # Iterate over each possible start time option (using 1-indexing).
        for i, duration in enumerate(delta_list, start=1):
            _option_counter += 1
            print(f"Loading option {_option_counter} of {_total_options} for intervention {intv_name}")
            try:
                option_workloads = {}
                for res_name, res_workload in workload_data.items():
                    # Initialize a list of zeros for absolute timesteps 1..T.
                    workload_list = [0] * instance.T
                    # For each absolute timestep from 1 to T:
                    for t in range(1, instance.T + 1):
                        # Check if this timestep is specified in the JSON.
                        if str(t) in res_workload:
                            inner_mapping = res_workload[str(t)]
                            # If the workload for this option's start time is provided, assign it.
                            if str(i) in inner_mapping:
                                workload_list[t - 1] = inner_mapping[str(i)]
                    option_workloads[res_name] = workload_list
            except Exception as e:
                logger.error(f"Error loading workload for intervention '{intv_name}', option {i}: {e}", exc_info=True)
                continue  # Skip this option if workload processing fails.
            
            try:
                # Initialize risks as a list of empty lists (one per absolute timestep).
                option_risks = [ [] for _ in range(instance.T) ]
                for t in range(1, instance.T + 1):
                    if str(t) in risk_data:
                        inner_mapping = risk_data[str(t)]
                        if str(i) in inner_mapping:
                            option_risks[t - 1] = inner_mapping[str(i)]
            except Exception as e:
                logger.error(f"Error loading risk for intervention '{intv_name}', option {i}: {e}", exc_info=True)
                continue  # Skip this option if risk processing fails.
            
            try:
                option = ScheduleOption(
                    start_time=i,
                    duration=duration,
                    workloads=option_workloads,
                    risks=option_risks
                )
                intervention.add_option(option)
            except Exception as e:
                logger.error(f"Error adding schedule option for intervention '{intv_name}', option {i}: {e}", exc_info=True)
                continue
        print(f"Finished loading options for intervention {intv_name}.")
        # Compute the overall mean risks list once all options are added.
        intervention.compute_overall_mean_risk()
        intervention.compute_mean_intervention_size()

        try:
            instance.add_intervention(intervention)
        except Exception as e:
            logger.error(f"Error adding intervention '{intv_name}': {e}", exc_info=True)
            continue
    print("Finished loading Interventions.")
    
    # Load Exclusions.
    print("Loading Exclusions...")
    _total_exclusions = len(json_data.get("Exclusions", {}))
    _excl_counter = 0
    for excl_key, excl_list in json_data.get("Exclusions", {}).items():
        _excl_counter += 1
        print(f"Loading exclusion {_excl_counter} of {_total_exclusions}: {excl_key}")
        try:
            if excl_list:
                season = excl_list[-1]
                interventions_excluded = excl_list[:-1]
                exclusion = Exclusion(
                    interventions=interventions_excluded,
                    season=season
                )
                instance.add_exclusion(exclusion)
        except Exception as e:
            logger.error(f"Error loading exclusion '{excl_key}': {e}", exc_info=True)
            continue
    print("Finished loading Exclusions.")
    
    print("Finished loading instance.")

    print("Creating risk and size clusters")
    try:
        instance.size_clusters = cluster_interventions_by_size(instance)
        instance.risk_clusters = cluster_interventions_by_risk(instance)
    except Exception as e:
            logger.error(f"Error clustering: {e}", exc_info=True)
    return instance


@dataclass
class Solution:
    """
    Represents a solution to the maintenance scheduling problem.
    
    Attributes:
        intervention_starts: Dictionary mapping intervention names to their start times
    """
    intervention_starts: Dict[str, int] = field(default_factory=dict)

    def __init__(self, solution_path: str):
        """
        Initialize solution from a text file.
        
        Args:
            solution_path: Path to solution file containing intervention start times
        """
        self.intervention_starts = {}
        try:
            with open(solution_path, 'r') as f:
                # Skip first line which is empty
                next(f)
                for line in f:
                    if line.strip():
                        intervention, start_time = line.strip().split()
                        # Remove 'Intervention_' prefix if present
                        intervention = intervention.replace('Intervention_', 'I')
                        self.intervention_starts[intervention] = int(start_time)
        except Exception as e:
            logger.error(f"Error loading solution from {solution_path}: {e}", exc_info=True)

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k,v in self.intervention_starts.items())
    
    def compute_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes and sets the concurrent_interventions and concurrency attributes for the solution.

        The concurrent_interventions attribute is a list of lists of length T (from the instance),
        where each inner list contains the names of interventions that are active at that timestep.
        The concurrency attribute is then computed as the length of each inner list.

        Args:
            instance (MaintenanceSchedulingInstance): The instance containing the planning horizon (T)
                and interventions.
        """
        # Initialize concurrent_interventions as a list of empty lists for each timestep.
        self.concurrent_interventions = [[] for _ in range(int(instance.T))]

        
        # Iterate over each scheduled intervention in the solution.
        for intervention_name, start_time in self.intervention_starts.items():
            # Retrieve the corresponding intervention object from the instance.
            intervention = instance.interventions.get(intervention_name)
            if intervention is None:
                logger.warning(f"Intervention {intervention_name} not found in instance.")
                continue
            
            # Find the schedule option matching the start_time.
            matching_option = None
            for option in intervention.options:
                if option.start_time == start_time:
                    matching_option = option
                    break
            if matching_option is None:
                logger.warning(f"No schedule option with start_time {start_time} for intervention {intervention_name}.")
                continue
            
            # Mark the intervention as active for its duration.
            # Note: timesteps and start times begin at 1; list indices start at 0.
            for t in range(start_time, start_time + matching_option.duration):
                #print(f"Duration: {matching_option.duration}, Start Time: {start_time}, Intervention: {intervention_name}, timestep append: {t}")
                if 1 <= t <= instance.T:
                    self.concurrent_interventions[t - 1].append(intervention_name)
        
        # Define concurrency as the number of interventions active at each timestep.
        self.concurrency = [len(interventions) for interventions in self.concurrent_interventions]
        self.highest_concurrency = max(self.concurrency)
    
    def compute_seansonality(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes and sets the seasonality and season interventions cluster attributes for the solution.

        The seasonality attribute is a dictionary mapping each season name (e.g., 'summer', 'winter', 'is')
        to the proportion of scheduled interventions that are active in that season.
        
        The season_interventions attribute is a dictionary with the same keys as seasonality,
        where each value is a list of interventions that are active in at least one timestep during that season.
        
        Additionally, creates a mapping from timestep to season (timestep_to_season) for use in plotting.
        
        Uses the concurrent_interventions attribute, and if it is not computed yet, runs compute_concurrency.
        
        Args:
            instance (MaintenanceSchedulingInstance): The instance containing the planning horizon (T) and seasons.
        """
        # Ensure concurrency data is available.
        if not hasattr(self, 'concurrent_interventions'):
            self.compute_concurrency(instance)
        
        total_interventions = len(self.intervention_starts)
        self.seasonality = {}         # e.g., {'summer': 0.4, 'winter': 0.3, 'is': 0.2}
        self.season_interventions = {} # e.g., {'summer': ['I1', 'I3'], 'winter': ['I2'], ...}
        
        # Iterate over each season defined in the instance.
        for season_name, season_obj in instance.seasons.items():
            active_interventions = set()
            # Loop over the periods (timesteps) belonging to the season.
            for period in season_obj.periods:
                try:
                    period_int = int(period)
                except ValueError:
                    logger.error(f"Period value '{period}' in season '{season_name}' is not an integer.")
                    continue
                if 1 <= period_int <= len(self.concurrent_interventions):
                    active_interventions.update(self.concurrent_interventions[period_int - 1])
            # Compute the proportion of scheduled interventions active during the season.
            prop = len(active_interventions) / total_interventions if total_interventions > 0 else 0
            self.seasonality[season_name] = prop
            self.season_interventions[season_name] = list(active_interventions)
        
        # Create a mapping from timestep to season name for plotting.
        self.timestep_to_season = {}
        for season_name, season_obj in instance.seasons.items():
            for period in season_obj.periods:
                try:
                    period_int = int(period)
                except ValueError:
                    logger.error(f"Period value '{period}' in season '{season_name}' is not an integer.")
                    continue
                if 1 <= period_int <= instance.T:
                    self.timestep_to_season[period_int] = season_name


    def plot_concurrency(self) -> None:
        """
        Creates a bar plot showing the number of concurrent interventions at each timestep.
        Bars are colored according to the season in which the timestep falls, if season data is available.
        If concurrency hasn't been computed yet, prompts user to run compute_concurrency first.
        """
        if not hasattr(self, 'concurrency'):
            print("Please run compute_concurrency() first to calculate concurrency data")
            return
        
        # If season data is available, assign colors based on season; otherwise, use a default.
        if not (hasattr(self, 'season_interventions') and hasattr(self, 'timestep_to_season')):
            colors = ['gray'] * len(self.concurrency)
        else:
            # Build a mapping of season name to color.
            seasons = list(self.season_interventions.keys())
            default_colors = ['gold', 'skyblue', 'lightgreen', 'salmon', 'plum']
            color_map = {'winter': 'lightblue', 'summer': 'peachpuff', 'is': 'lightgreen'}
            
            # Assign a color for each timestep based on its season.
            colors = []
            for t in range(1, len(self.concurrency) + 1):
                season = self.timestep_to_season.get(t, None)
                colors.append(color_map.get(season, 'gray'))
        
        plt.figure(figsize=(10,6))
        plt.bar(range(1, len(self.concurrency) + 1), self.concurrency, color=colors)
        plt.xlabel('Timestep')
        plt.ylabel('Number of Concurrent Interventions')
        plt.title('Intervention Concurrency Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add legend if season data is available.
        if hasattr(self, 'season_interventions'):
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=color_map[season], label=season) for season in color_map]
            plt.legend(handles=patches, title="Season")
            
        plt.show()


    def set_worst_risks(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Use self.concurrent_interventions together with the dictionary of distance matrices (intervention-intervention) where each matrix corresponds to the membership values to "close", "mid-close", "mid", "mid-far", "far" (values are these strings, keys are the dataframes that correspond to the matrices) 
        For each day, take the submatrix containing only the rows and columns corresponding to the intervetions of that day.
        Then, the submatrix is converted into a mass, which corresponds to the cardinality of that fuzzy relation, by adding all the entries of the triangular upper or lower submatrix without the diagonal (memberships of unique pairs).
        
        This gives us a vector of length T and entries the cardinality of the submatrix of each day.
        This vector is normalized dividing by the total mass and converted into a scalar by computing its entropy (take absolute value, I want it positive because it is a measure of the deviation from the uniform distribution).
        Finally that scalar is normalized dividing by the entropy of the uniform distribution.

        In the end we get a scalar value for each membership. This will be stored in self.closeness_concurrency, which will be a dictionary with the same keys as dist_mat and with values those scalars.
        """
        self.worst_risks = []
        self.intervention_worst_risk = {}
        for intervention_name, start_time in self.intervention_starts.items():
            # Retrieve the corresponding intervention object from the instance.
            intervention = instance.interventions.get(intervention_name)
            if intervention is None:
                logger.warning(f"Intervention {intervention_name} not found in instance.")
                continue
            
            # Find the schedule option matching the start_time.
            matching_option = None
            for option in intervention.options:
                if option.start_time == start_time:
                    matching_option = option
                    break
            if matching_option is None:
                logger.warning(f"No schedule option with start_time {start_time} for intervention {intervention_name}.")
                continue
            
            worst_risk = matching_option.worst_risk
            self.worst_risks.append(worst_risk)
            self.intervention_worst_risk[intervention_name] = worst_risk
        
        self.highest_risk = sum(self.worst_risks) / len(self.worst_risks) # Taking the mean across all instances' highest risks.

    def dist_matrix_to_closeness_concurrency(self, dist_mat: Dict[str, 'pd.DataFrame']) -> None:  
        """Use self.concurrent_interventions together with the dictionary of distance matrices (intervention-intervention) where each matrix corresponds to the membership values to "close", "mid-close", "mid", "mid-far", "far" (values are these strings, keys are the dataframes that correspond to the matrices) 
        For each day, take the submatrix containing only the rows and columns corresponding to the intervetions of that day.
        Then, the submatrix is converted into a mass, which corresponds to the cardinality of that fuzzy relation, by adding all the entries of the triangular upper or lower submatrix without the diagonal (memberships of unique pairs).
        
        This gives us a vector of length T and entries the cardinality of the submatrix of each day.
        This vector is normalized dividing by the total mass and converted into a scalar by computing its entropy (take absolute value, I want it positive because it is a measure of the deviation from the uniform distribution).
        Finally that scalar is normalized dividing by the entropy of the uniform distribution.

        In the end we get a scalar value for each membership. This will be stored in self.closeness_concurrency, which will be a dictionary with the same keys as dist_mat and with values those scalars."""

        if not hasattr(self, 'concurrent_interventions'):
            raise RuntimeError("Run `compute_concurrency()` first so that `self.concurrent_interventions` is available.")

        T = len(self.concurrent_interventions)
        terms = list(dist_mat.keys())
        # --- Prepare an empty mass vector per fuzzy term ----------------
        mass: Dict[str, np.ndarray] = {term: np.zeros(T, dtype=float) for term in terms}

        # --- 1‑B.  Accumulate masses per timestep -----------------------------
        for t, active in enumerate(self.concurrent_interventions):
            if len(active) < 2:          # ≤ 1 intervention → no pair to measure
                continue
            for term, df in dist_mat.items():
                # restrict to the active interventions *in the same order* as df
                sub = df.loc[active, active]
                # keep upper‑triangular part (excluding diagonal)
                tri_mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
                tri_sum = sub.values[tri_mask].sum()
                mass[term][t] = tri_sum

        # --- 1‑C.  Entropy‑based scalar per term -----------------------------
        self.closeness_concurrency: Dict[str, float] = {}
        for term, vec in mass.items():
            total = vec.sum()
            if total == 0:
                self.closeness_concurrency[term] = 0.0
                continue
            p = vec / total
            nz = p > 0                    # avoid log(0)
            H = -np.sum(p[nz] * np.log(p[nz]))
            # Entropy of the *effective* uniform distribution (non‑zero entries)
            H_unif = math.log(nz.sum()) if nz.any() else 1.0
            self.closeness_concurrency[term] = H / H_unif if H_unif > 0 else 0.0

        # Save daily masses for plotting purposes
        self._closeness_daily_mass = mass

    def compute_environmental_impact_concurrency(self, dist_mat: Dict[str, 'pd.DataFrame'], tconorm) -> None:  
        """Use self.concurrent_interventions together with the dictionary of distance matrices (intervention(rows)-park(cols)) where each matrix corresponds to the membership values to "close", "mid-close", "mid", "mid-far", "far" (values are these strings, keys are the dataframes that correspond to the matrices) 
        Collapse the matrix into a vector by applying the t-conorm to each row, so that we get a unique value for each intervention (you can use tconorm_aggregate from fuzzy_var)
        For each day, take the subvector containing only the rows corresponding to the intervetions of that day.
        Then, the subvector is converted into a mass, which corresponds to the cardinality of that fuzzy set, by adding all the entries (memberships).
        
        This gives us a vector of length T and entries the cardinality of the subvectors of each day.
        This vector is normalized dividing by the total mass and converted into a scalar by computing its entropy (take absolute value, I want it positive because it is a measure of the deviation from the uniform distribution).
        Finally that scalar is normalized dividing by the entropy of the uniform distribution.

        In the end we get a scalar value for each membership. This will be stored in self.closeness_concurrency, which will be a dictionary with the same keys as dist_mat and with values those scalars."""

        if not hasattr(self, 'concurrent_interventions'):
            raise RuntimeError("Run `compute_concurrency()` first so that `self.concurrent_interventions` is available.")

        # --- 2‑A.  Collapse each membership matrix → vector (length = n_interventions)
        row_aggs: Dict[str, pd.Series] = {}
        for term, df in dist_mat.items():
            agg = tconorm_aggregate(df, tconorm)            # numpy array
            row_aggs[term] = pd.Series(agg, index=df.index)  # Series: μ(term | intervention)

        T = len(self.concurrent_interventions)
        mass: Dict[str, np.ndarray] = {term: np.zeros(T, dtype=float) for term in dist_mat}

        # --- 2‑B.  Mass per timestep ----------------------------------------
        for t, active in enumerate(self.concurrent_interventions):
            if not active:
                continue
            for term, series in row_aggs.items():
                mass[term][t] = series.loc[active].sum()
        
        # Save daily masses for plotting purposes
        self._env_impact_daily_mass = mass

        # --- 2‑C.  Entropy‑based scalar per term -----------------------------
        self.environmental_impact_concurrency: Dict[str, float] = {}
        for term, vec in mass.items():
            total = vec.sum()
            if total == 0:
                self.environmental_impact_concurrency[term] = 0.0
                continue
            p = vec / total
            nz = p > 0
            H = -np.sum(p[nz] * np.log(p[nz]))
            H_unif = math.log(nz.sum()) if nz.any() else 1.0
            self.environmental_impact_concurrency[term] = H / H_unif if H_unif > 0 else 0.0

    def compute_size_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes size concurrency for the solution based on the clustering of interventions by mean size.
        
        It uses the cluster_interventions_by_size function (with labels 'small', 'mid', 'big') to
        determine which interventions are in the 'big' and 'mid' clusters.
        For each timestep, the score is computed as:
            - (n-1)**2 for interventions in the 'big' cluster (only if n > 0)
            - (m-1) for interventions in the 'mid' cluster (only if m > 0)
        The per-timestep lists are stored in self.big_size_concurrent_interventions and 
        self.mid_size_concurrent_interventions, and the overall score in self.size_concurrency.
        
        Args:
            instance (MaintenanceSchedulingInstance): The instance containing the interventions.
        """
        # Ensure concurrency data is available.
        if not hasattr(self, 'concurrent_interventions'):
            print("Please run compute_concurrency() first to calculate concurrency data")
            return

        # Compute clustering dictionary for sizes (keys: 'small', 'mid', 'big')
        size_clusters = instance.size_clusters

        self.big_size_concurrent_interventions = []
        self.mid_size_concurrent_interventions = []
        self.size_concurrency = 0.0

        for interventions in self.concurrent_interventions:
            # Filter interventions that are in the 'big' and 'mid' clusters.
            big_list = [i for i in interventions if i in size_clusters.get('big', [])]
            mid_list = [i for i in interventions if i in size_clusters.get('mid', [])]
            
            self.big_size_concurrent_interventions.append(big_list)
            self.mid_size_concurrent_interventions.append(mid_list)
            
            n = len(big_list)
            m = len(mid_list)
            # Score: (n-1)**2 for big and (m-1) for mid (only if count > 0)
            score_big = (max(n - 1, 0))**2
            score_mid = max(m - 1, 0)
            self.size_concurrency += score_big + score_mid


    def compute_risk_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes risk concurrency for the solution based on the clustering of interventions by average risk.
        
        It uses the cluster_interventions_by_risk function (with labels 'low', 'mid', 'high') to
        determine which interventions fall into the 'high' and 'mid' risk clusters.
        For each timestep, the score is computed as:
            - (n-1)**2 for interventions in the 'high' cluster (only if n > 0)
            - (m-1) for interventions in the 'mid' cluster (only if m > 0)
        The per-timestep lists are stored in self.high_risk_concurrent_interventions and 
        self.mid_risk_concurrent_interventions, and the overall score in self.risk_concurrency.
        
        Args:
            instance (MaintenanceSchedulingInstance): The instance containing the interventions.
        """
        # Ensure concurrency data is available.
        if not hasattr(self, 'concurrent_interventions'):
            print("Please run compute_concurrency() first to calculate concurrency data")
            return

        # Compute clustering dictionary for risks (keys: 'low', 'mid', 'high')
        risk_clusters = instance.risk_clusters

        self.high_risk_concurrent_interventions = []
        self.mid_risk_concurrent_interventions = []
        self.risk_concurrency = 0.0

        for interventions in self.concurrent_interventions:
            # Filter interventions that are in the 'high' and 'mid' risk clusters.
            high_list = [i for i in interventions if i in risk_clusters.get('high', [])]
            mid_list = [i for i in interventions if i in risk_clusters.get('mid', [])]
            
            self.high_risk_concurrent_interventions.append(high_list)
            self.mid_risk_concurrent_interventions.append(mid_list)
            
            n = len(high_list)
            m = len(mid_list)
            # Score: (n-1)**2 for high and (m-1) for mid (only if count > 0)
            score_high = (max(n - 1, 0))**2
            score_mid = max(m - 1, 0)
            self.risk_concurrency += score_high + score_mid


    def plot_all_concurrency_details(self) -> None:
        """
        Creates informative plots of the concurrency levels of size, environmental impact,
        risk, and closeness at each timestep, with background colored boxes indicating the season
        for each contiguous chunk of timesteps (using self.timestep_to_season).

        For each attribute, two grouped bar charts (one for the "high" group and one for the "mid" group)
        are plotted in a 2x2 grid of subplots. The background for each subplot is annotated with
        season labels using the same color palette as in plot_concurrency.
        """
        # Ensure that concurrency data is available.
        if not hasattr(self, 'concurrent_interventions'):
            print("Please run compute_concurrency() first to calculate concurrency data.")
            return

        T = len(self.concurrent_interventions)
        timesteps = np.arange(1, T + 1)
        width = 0.35  # width of the grouped bars

        # Prepare counts for each attribute based on the per-timestep lists.
        size_high     = np.array([len(lst) for lst in self.big_size_concurrent_interventions])
        size_mid      = np.array([len(lst) for lst in self.mid_size_concurrent_interventions])

        env_high      = np.array([len(lst) for lst in self.high_env_imp_concurrent_interventions])
        env_mid       = np.array([len(lst) for lst in self.mid_env_imp_concurrent_interventions])

        risk_high     = np.array([len(lst) for lst in self.high_risk_concurrent_interventions])
        risk_mid      = np.array([len(lst) for lst in self.mid_risk_concurrent_interventions])

        closeness_high = np.array([len(lst) for lst in self.close_concurrent_interventions])
        closeness_mid  = np.array([len(lst) for lst in self.mid_concurrent_interventions])

        # Compute contiguous season intervals from self.timestep_to_season.
        # Each interval is a tuple: (season, start_timestep, end_timestep)
        season_intervals = []
        current_season = None
        start = None
        for t in range(1, T + 1):
            season = self.timestep_to_season.get(t, None)
            if season != current_season:
                if current_season is not None:
                    season_intervals.append((current_season, start, t - 1))
                current_season = season
                start = t
        if current_season is not None:
            season_intervals.append((current_season, start, T))

        # Define the season color palette (same as in plot_concurrency).
        color_map = {'winter': 'lightblue', 'summer': 'peachpuff', 'is': 'lightgreen'}

        def add_season_background(ax):
            """
            For the given axis, add a background rectangle (using axvspan) for each season interval.
            The rectangle covers from (start - 0.5) to (end + 0.5) in x-coordinates.
            Also, adds the season label at the center of the rectangle.
            """
            for season, s, e in season_intervals:
                if season is None:
                    continue
                xmin = s - 0.5
                xmax = e + 0.5
                # Draw a background rectangle with a low alpha so that the bars remain visible.
                ax.axvspan(xmin, xmax, facecolor=color_map.get(season, 'gray'), alpha=0.5, zorder=0)
                # Place the season label in the center of the interval.
                x_text = (xmin + xmax) / 2
                ylim = ax.get_ylim()
                # Position the text near the top of the plotting area.
                y_text = ylim[1] - (ylim[1] - ylim[0]) * 0.05
                ax.text(x_text, y_text, season, ha='center', va='top', fontsize=8, color='black', zorder=3)

        # Create a 2x2 grid of subplots.
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Detailed Concurrency Levels per Timestep with Seasons", fontsize=16)

        # Plot Size Concurrency.
        ax = axes[0, 0]
        ax.bar(timesteps - width/2, size_high, width, label='Big', color='blue', zorder=2)
        ax.bar(timesteps + width/2, size_mid, width, label='Mid', color='orange', zorder=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.set_title("Size Concurrency")
        add_season_background(ax)
        # Place legend outside the plot area so it doesn't overlap the season text.
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # Plot Environmental Impact Concurrency.
        ax = axes[0, 1]
        ax.bar(timesteps - width/2, env_high, width, label='High Env Impact', color='blue', zorder=2)
        ax.bar(timesteps + width/2, env_mid, width, label='Mid Env Impact', color='orange', zorder=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.set_title("Environmental Impact Concurrency")
        add_season_background(ax)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # Plot Risk Concurrency.
        ax = axes[1, 0]
        ax.bar(timesteps - width/2, risk_high, width, label='High Risk', color='blue', zorder=2)
        ax.bar(timesteps + width/2, risk_mid, width, label='Mid Risk', color='orange', zorder=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.set_title("Risk Concurrency")
        add_season_background(ax)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # Plot Closeness Concurrency.
        ax = axes[1, 1]
        ax.bar(timesteps - width/2, closeness_high, width, label='Close Pairs', color='blue', zorder=2)
        ax.bar(timesteps + width/2, closeness_mid, width, label='Mid Pairs', color='orange', zorder=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count")
        ax.set_title("Closeness Concurrency")
        add_season_background(ax)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        plt.show()



# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    #Load Json
    json_path = 'Decision Matrix/Problem setups/C_01.json'
    with open(json_path, "r") as f:
        data = json.load(f)

    # Example JSON (could be loaded from a file) in the original format:
    example_json = {
        "T": 7,
        "Scenarios_number": [3, 2, 3], # Numero de escenarios para cada t, lista de longitud T
        "Resources": {
            "c1": {
                "min": [10.0, 0.0, 6.0],
                "max": [49.0, 23.0, 15.0]
            }
        },
        "Seasons": {
            "winter": [1, 2],
            "summer": [3],
            "is": []
        },
        "Interventions": {
            "I1": {
                "tmax": 1,
                "Delta": [3, 3, 2], #Duration depending on the start time.
                "workload": {
                    "c1": {
                        "2": {"1": 14, "2": 0, "3": 1}, #tstep: {st : workload needed of that resource}
                        "3": {"1": 0, "2": 14, "3": 1},
                        "4": {"1": 0, "2": 0, "3": 14}
                    }
                }, 
                "risk": {
                    "2": {"1": [4, 8, 2], "2": [0, 0, 0]}, #tstep:{st : risk for each scenario (list of length scenario number for that timestep)}
                    "3": {"1": [0, 0], "2": [3, 8]},
                    "4": {"1": [0, 0, 0], "2": [0, 0, 0]}
                }
            }
        },
        "Exclusions": {
            "E1": ["I2", "I3", "full"]
        }
    }
    
    # Load instance from JSON.
    instance = load_instance_from_json(data)

    #print(instance.interventions.get("I135").name)
    #print(f"# interventions: {len(instance.interventions)}")

    # Print the instance with detailed metrics.
    #instance.show()

    # Test loading solution from file
    solution_path = 'Decision Matrix/Alternatives/1/solution_C_01_900.txt'
    
    sol = Solution(solution_path)
    
    print(max(sol.intervention_starts.values()))

    sol.compute_concurrency(instance)


    # print(f"Concurrency:\n{sol.concurrency}")
    sol.plot_concurrency()
    #print(f"\n\nConcurrent Interventions:\n{sol.concurrent_interventions}")
    sol.compute_seansonality(instance)

    print(sol.seasonality) #! Does not add to 1 since there are multiple interventions spanning various seasons.
    sol.plot_concurrency()

    sol.set_worst_risks(instance)

    print(sol.worst_risks)

    # Print mean intervention sizes
    # print("\nIntervention mean sizes:")
    # for intervention in instance.interventions.values():
    #     intervention.compute_mean_intervention_size()
    #     print(f"{intervention.name}: {intervention.mean_intervention_size:.2f}")
