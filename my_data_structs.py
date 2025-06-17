import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging
import itertools
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans # Keep for reference or other potential uses
from fuzzy_var import tconorm_aggregate


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

def fuzzy_cluster_by_attribute(values: List[float], names: List[str], labels: List[str]) -> pd.DataFrame:
    """
    Generic fuzzy clustering function that applies fuzzy c-means (c=5) to 1D values,
    and maps the clusters (in ascending order of centroid value) to the provided labels.
    The result is a DataFrame of membership values.
    
    Args:
        values: A list of one-dimensional real values.
        names: A list of instance names corresponding to the values.
        labels: A list of 5 cluster labels ordered from smallest to largest.
                e.g., for sizes: ['small', 'mid-small', 'medium', 'mid-large', 'large'].
                
    Returns:
        A pandas DataFrame where rows are instance names and columns are cluster labels,
        containing the membership value of each instance to each cluster.
    """
    if len(set(values)) < len(labels):
        logger.warning("Number of unique values is less than number of clusters. Clustering may be ineffective.")
    
    data = np.array(values).reshape(-1, 1).T  # Fuzzy c-means expects features in rows
    
    # Apply fuzzy c-means
    # n_clusters=len(labels) to be generic, but request specifies 5
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, c=len(labels), m=2, error=0.005, maxiter=1000, init=None, seed=42
    )
    
    # `u` has shape (n_clusters, n_samples), transpose it
    memberships = u.T
    
    # Sort centroids to map to labels correctly (smallest centroid -> labels[0], etc.)
    sorted_centroid_indices = np.argsort(cntr.flatten())
    
    # Reorder the membership columns to match the sorted labels
    sorted_memberships = memberships[:, sorted_centroid_indices]
    
    # Informative print of the centroids for each label
    print("Fuzzy Cluster Centroids:")
    for i, label in enumerate(labels):
        centroid_value = cntr[sorted_centroid_indices[i]]
        print(f"  {label}: {centroid_value[0]:.4f}")
        
    # Create a DataFrame for easy lookup
    membership_df = pd.DataFrame(sorted_memberships, index=names, columns=labels)
    
    return membership_df


def cluster_interventions_by_size(instance) -> pd.DataFrame:
    """
    Clusters interventions based on mean intervention size using fuzzy c-means.
    Uses 5 labels: 'small', 'mid-small', 'medium', 'mid-large', 'large'.
    
    Returns:
        A pandas DataFrame of membership values (rows: interventions, cols: size labels).
    """
    names = []
    sizes = []
    for intervention in instance.interventions.values():
        size = intervention.mean_intervention_size
        if size == 0.0:
            size = intervention.compute_mean_intervention_size()
        names.append(intervention.name)
        sizes.append(size)
        
    labels = ['small', 'mid-small', 'medium', 'mid-large', 'large']
    print("\nClustering interventions by size...")
    return fuzzy_cluster_by_attribute(sizes, names, labels)

def cluster_interventions_by_risk(instance) -> pd.DataFrame:
    """
    Clusters interventions based on average overall risk using fuzzy c-means.
    Uses 5 labels: 'low', 'mid-low', 'mid', 'mid-high', 'high'.

    Returns:
        A pandas DataFrame of membership values (rows: interventions, cols: risk labels).
    """
    names = []
    avg_risks = []
    for intervention in instance.interventions.values():
        if not intervention.overall_mean_risk:
            intervention.compute_overall_mean_risk()
        avg_risk = sum(intervention.overall_mean_risk) / len(intervention.overall_mean_risk) if intervention.overall_mean_risk else 0.0
        names.append(intervention.name)
        avg_risks.append(avg_risk)
        
    labels = ['low', 'mid-low', 'mid', 'mid-high', 'high']
    print("\nClustering interventions by risk...")
    return fuzzy_cluster_by_attribute(avg_risks, names, labels)


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
        size_memberships: DataFrame of fuzzy cluster memberships for intervention sizes.
        risk_memberships: DataFrame of fuzzy cluster memberships for intervention risks.
    """
    T: int
    scenarios_number: List[int]
    resources: Dict[str, Resource] = field(default_factory=dict)
    seasons: Dict[str, Season] = field(default_factory=dict)
    interventions: Dict[str, Intervention] = field(default_factory=dict)
    exclusions: List[Exclusion] = field(default_factory=list)
    size_memberships: 'pd.DataFrame' = field(init=False, default=None)
    risk_memberships: 'pd.DataFrame' = field(init=False, default=None)
    
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

# ------------------------------------------------------------------------------
# Function to load from the original JSON structure into the new object model
# ------------------------------------------------------------------------------

def load_instance_from_json(json_data: Dict[str, Any]) -> MaintenanceSchedulingInstance:
    """
    Load a MaintenanceSchedulingInstance from a JSON-like dictionary.
    This function populates the data classes and, upon completion, automatically
    runs fuzzy c-means clustering for intervention 'size' and 'risk', storing
    the membership dataframes in the instance object.

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
        #print(f"Loading resource {_res_counter} of {_total_resources}: {res_name}")
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
        #print(f"Loading season {_season_counter} of {_total_seasons}: {season_name}")
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
        #print(f"Loading intervention {_intv_counter} of {_total_interventions}: {intv_name}")
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
            #print(f"Loading option {_option_counter} of {_total_options} for intervention {intv_name}")
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
        #print(f"Finished loading options for intervention {intv_name}.")
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
        #print(f"Loading exclusion {_excl_counter} of {_total_exclusions}: {excl_key}")
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

    print("Creating fuzzy cluster memberships for size and risk...")
    try:
        instance.size_memberships = cluster_interventions_by_size(instance)
        instance.risk_memberships = cluster_interventions_by_risk(instance)
    except Exception as e:
        logger.error(f"Error during fuzzy clustering: {e}", exc_info=True)
    return instance


@dataclass
class Solution:
    """
    Represents a solution to the maintenance scheduling problem.
    """
    intervention_starts: Dict[str, int] = field(default_factory=dict)

    def __init__(self, solution_path: str):
        """
        Initialize solution from a text file.
        """
        self.intervention_starts = {}
        try:
            with open(solution_path, 'r') as f:
                next(f)
                for line in f:
                    if line.strip():
                        intervention, start_time = line.strip().split()
                        intervention = intervention.replace('Intervention_', 'I')
                        self.intervention_starts[intervention] = int(start_time)
        except Exception as e:
            logger.error(f"Error loading solution from {solution_path}: {e}", exc_info=True)

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k,v in self.intervention_starts.items())
    
    def compute_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes the list of active interventions for each timestep.
        """
        self.concurrent_interventions = [[] for _ in range(int(instance.T))]
        for intervention_name, start_time in self.intervention_starts.items():
            intervention = instance.interventions.get(intervention_name)
            if not intervention: continue
            matching_option = next((opt for opt in intervention.options if opt.start_time == start_time), None)
            if not matching_option: continue
            for t in range(start_time, start_time + matching_option.duration):
                if 1 <= t <= instance.T:
                    self.concurrent_interventions[t - 1].append(intervention_name)
        self.concurrency = [len(interventions) for interventions in self.concurrent_interventions]
        self.highest_concurrency = max(self.concurrency)
    
    def compute_seansonality(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes seasonality metrics for the solution.
        """
        if not hasattr(self, 'concurrent_interventions'): self.compute_concurrency(instance)
        total_interventions = len(self.intervention_starts)
        self.seasonality, self.season_interventions = {}, {}
        for name, season in instance.seasons.items():
            active_in_season = set()
            for period in season.periods:
                if 1 <= int(period) <= len(self.concurrent_interventions):
                    active_in_season.update(self.concurrent_interventions[int(period) - 1])
            self.seasonality[name] = len(active_in_season) / total_interventions if total_interventions > 0 else 0
            self.season_interventions[name] = list(active_in_season)
        self.timestep_to_season = {int(p): name for name, s in instance.seasons.items() for p in s.periods}


    def plot_concurrency(self) -> None:
        """
        Creates a bar plot of concurrent interventions, colored by season.
        """
        if not hasattr(self, 'concurrency'):
            print("Please run compute_concurrency() first.")
            return
        
        color_map = {'winter': 'lightblue', 'summer': 'peachpuff', 'is': 'lightgreen'}
        colors = [color_map.get(self.timestep_to_season.get(t), 'gray') for t in range(1, len(self.concurrency) + 1)]
        
        plt.figure(figsize=(10,6))
        plt.bar(range(1, len(self.concurrency) + 1), self.concurrency, color=colors)
        plt.xlabel('Timestep'); plt.ylabel('Number of Concurrent Interventions'); plt.title('Intervention Concurrency Over Time')
        plt.grid(True, alpha=0.3)
        
        if hasattr(self, 'season_interventions'):
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=color_map[s], label=s) for s in color_map if s in self.season_interventions]
            plt.legend(handles=patches, title="Season")
        plt.show()

    def set_worst_risks(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Calculates the mean of the worst-case risks for each scheduled intervention.
        """
        self.worst_risks = []
        self.intervention_worst_risk = {}
        for int_name, start_time in self.intervention_starts.items():
            intervention = instance.interventions.get(int_name)
            if not intervention: continue
            option = next((opt for opt in intervention.options if opt.start_time == start_time), None)
            if not option: continue
            self.worst_risks.append(option.worst_risk)
            self.intervention_worst_risk[int_name] = option.worst_risk
        self.highest_risk = sum(self.worst_risks) / len(self.worst_risks) if self.worst_risks else 0.0

    def _compute_entropy_score_from_mass(self, mass_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Return a normalized entropy score in [0, 1] for each mass vector.

        1 → vector is uniform (even an all-zero vector counts as uniform)  
        0 → vector is a Dirac (all the mass in one day)
        """
        scores = {}
        for term, vec in mass_dict.items():
            total = vec.sum()

            # --- all-zero vector → score 1 (perfectly uniform) -------------
            if total == 0:
                scores[term] = 1.0
                continue

            p  = vec / total
            nz = p > 0                            # avoid log(0)
            H  = -np.sum(p[nz] * np.log(p[nz]))   # Shannon entropy
            H_unif = math.log(nz.sum())           # entropy of the effective uniform
            scores[term] = H / H_unif if H_unif else 0.0

        return scores

    # def _compute_entropy_score_from_mass(self, mass_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    #     """
    #     Uniformity score based on Jensen-Shannon divergence.

    #         JS(p || u) = ½·KL(p || m) + ½·KL(u || m),   m = ½(p + u)
    #         score      = 1 - JS(p || u) / ln(2)

    #     • u is the uniform distribution over the *full* vector length T.
    #     • score ∈ [0, 1]      (natural logs ⇒ JS ≤ ln 2)
    #         - 1 → perfectly uniform (or all-zero) vector
    #         - 0 → Dirac vector (all mass in one day)
    #     """
    #     ln2 = math.log(2.0)
    #     scores: Dict[str, float] = {}

    #     for term, vec in mass_dict.items():
    #         T = len(vec)
    #         if T <= 1:
    #             scores[term] = 1.0
    #             continue

    #         total = vec.sum()
    #         if total == 0:                   # all-zero vector → treat as uniform
    #             scores[term] = 1.0
    #             continue

    #         p = vec / total
    #         u = np.full(T, 1.0 / T)          # fixed uniform reference
    #         m = 0.5 * (p + u)

    #         # KL divergences, skipping 0·log0 terms
    #         kl_pm = np.where(p > 0, p * np.log(p / m), 0.0).sum()
    #         kl_um = (u * np.log(u / m)).sum()          # u has no zeros

    #         js = 0.5 * (kl_pm + kl_um)
    #         scores[term] = max(0.0, 1.0 - js / ln2)    # numerical safety

    #     return scores

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

        # --- Accumulate masses per timestep -----------------------------
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

        # Save daily masses for plotting purposes
        self._closeness_daily_mass = mass

        # Normalised entropy score (deviation from uniform)
        self.closeness_concurrency = self._compute_entropy_score_from_mass(mass)

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

        # --- Collapse each membership matrix → vector (length = n_interventions)
        row_aggs: Dict[str, pd.Series] = {}
        for term, df in dist_mat.items():
            agg = tconorm_aggregate(df, tconorm)            # numpy array
            row_aggs[term] = pd.Series(agg, index=df.index)  # Series: μ(term | intervention)

        T = len(self.concurrent_interventions)
        mass: Dict[str, np.ndarray] = {term: np.zeros(T, dtype=float) for term in dist_mat}

        # --- Mass per timestep ----------------------------------------
        for t, active in enumerate(self.concurrent_interventions):
            if not active:
                continue
            for term, series in row_aggs.items():
                mass[term][t] = series.loc[active].sum()
        
        # Save daily masses for plotting purposes
        self._env_impact_daily_mass = mass

        # Normalised entropy score (deviation from uniform)
        self.environmental_impact_concurrency = self._compute_entropy_score_from_mass(mass)

    def compute_size_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes size concurrency based on fuzzy c-means memberships.

        For each of the 5 size clusters (e.g., 'small', 'large'), it calculates a daily "mass"
        by summing the membership values of all active interventions for that day. This results
        in a daily mass vector for each cluster. These vectors are stored in `_size_daily_mass`
        for plotting.

        Finally, it computes a single scalar score for each cluster's mass vector using a normalized
        entropy measure, which indicates how evenly the concurrency mass is distributed over time.
        These scores are stored in the `size_concurrency` dictionary.
        """
        if not hasattr(self, 'concurrent_interventions'): self.compute_concurrency(instance)
        if instance.size_memberships is None:
            raise ValueError("Size memberships not found in instance. Run `load_instance_from_json` first.")

        T = len(self.concurrent_interventions)
        memberships_df = instance.size_memberships
        labels = memberships_df.columns.tolist()
        
        mass = {label: np.zeros(T) for label in labels}
        for t, active_interventions in enumerate(self.concurrent_interventions):
            if not active_interventions:
                continue
            # Ensure all active interventions are in the membership index
            valid_active = [iv for iv in active_interventions if iv in memberships_df.index]
            if not valid_active:
                continue

            daily_masses = memberships_df.loc[valid_active].sum(axis=0)
            for label in labels:
                mass[label][t] = daily_masses[label]

        self._size_daily_mass = mass
        self.size_concurrency = self._compute_entropy_score_from_mass(mass)


    def compute_risk_concurrency(self, instance: MaintenanceSchedulingInstance) -> None:
        """
        Computes risk concurrency based on fuzzy c-means memberships.

        This method mirrors `compute_size_concurrency` but uses the risk cluster memberships.
        It calculates a daily mass for each of the 5 risk clusters ('low', 'high', etc.)
        and stores them in `_risk_daily_mass`. It then computes a normalized entropy score
        for each cluster, storing the results in the `risk_concurrency` dictionary.
        """
        if not hasattr(self, 'concurrent_interventions'): self.compute_concurrency(instance)
        if instance.risk_memberships is None:
            raise ValueError("Risk memberships not found in instance. Run `load_instance_from_json` first.")

        T = len(self.concurrent_interventions)
        memberships_df = instance.risk_memberships
        labels = memberships_df.columns.tolist()

        mass = {label: np.zeros(T) for label in labels}
        for t, active_interventions in enumerate(self.concurrent_interventions):
            if not active_interventions:
                continue
            valid_active = [iv for iv in active_interventions if iv in memberships_df.index]
            if not valid_active:
                continue
            
            daily_masses = memberships_df.loc[valid_active].sum(axis=0)
            for label in labels:
                mass[label][t] = daily_masses[label]

        self._risk_daily_mass = mass
        self.risk_concurrency = self._compute_entropy_score_from_mass(mass)

    def plot_all_concurrency_details(self) -> None:
        """Creates a 2x2 grid of stacked bar plots for all four concurrency masses."""
        required_attrs = ['_size_daily_mass', '_risk_daily_mass', '_closeness_daily_mass', '_env_impact_daily_mass']
        if not all(hasattr(self, attr) for attr in required_attrs):
            print("Please run all four concurrency calculation methods first.")
            return
        if not hasattr(self, 'timestep_to_season'):
             print("Please run `compute_seansonality()` first.")
             return

        fig, axes = plt.subplots(2, 2, figsize=(19, 14), sharex=True)
        fig.suptitle("Daily Concurrency Mass by Fuzzy Category", fontsize=16)
        T = len(self.concurrent_interventions)
        timesteps = np.arange(1, T + 1)
        
        color_map_seasons = {'winter': 'lightblue', 'summer': 'peachpuff', 'is': 'lightgreen'}
        season_intervals = []
        if self.timestep_to_season:
            for season, group in itertools.groupby(sorted(self.timestep_to_season.items()), key=lambda item: item[1]):
                timesteps_in_season = [item[0] for item in group]
                season_intervals.append((season, min(timesteps_in_season), max(timesteps_in_season)))

        def add_season_background(ax):
            y_lim = ax.get_ylim()
            for season, start, end in season_intervals:
                ax.axvspan(start - 0.5, end + 0.5, facecolor=color_map_seasons.get(season, 'gray'), alpha=0.3, zorder=0)
                ax.text((start + end) / 2, y_lim[1] *1.05, season, ha='center', va='top', fontsize=9)
            ax.set_ylim(y_lim[0], y_lim[1] * 1.1) # Reset ylim after adding text with extra space

        def plot_stacked_bar(ax, mass_data, title, log = False):
            bottom = np.zeros(T)
            for label, mass_vector in mass_data.items():
                ax.bar(timesteps, mass_vector, bottom=bottom, label=label, zorder=2)
                bottom += mass_vector
            ax.set_ylabel("Total Membership (Mass)")
            ax.set_xlabel("Timestep")
            if log: 
                ax.set_yscale('log')
            ax.set_title(title)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            add_season_background(ax)

        plot_stacked_bar(axes[0, 0], self._size_daily_mass, "Size Concurrency Mass")
        plot_stacked_bar(axes[0, 1], self._risk_daily_mass, "Risk Concurrency Mass")
        plot_stacked_bar(axes[1, 0], self._closeness_daily_mass, "Closeness Concurrency Mass", log=True)
        plot_stacked_bar(axes[1, 1], self._env_impact_daily_mass, "Environmental Impact Concurrency Mass")

        plt.tight_layout(rect=[0, 0, 0.9, 0.96], h_pad=3.0)
        plt.show()


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    json_path = 'Decision Matrix\Difficult Instances\X_12.json' 
    solution_path = 'Decision Matrix\Alternatives\X12\Team1\sol300_t1_X_12_s42.txt' 
    
    # 1. Load Instance JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        instance = load_instance_from_json(data)
    except FileNotFoundError:
        print(f"ERROR: Instance file not found at '{json_path}'. Exiting.")
        exit()

    # 2. Load Solution
    sol = Solution(solution_path)
    if not sol.intervention_starts:
        print("ERROR: Solution could not be loaded or is empty. Exiting.")
        exit()

    # 3. Load Fuzzy Distance Matrices
    import pickle
    try:
        with open('interv_mems.pkl', 'rb') as f:
            interv_mems = pickle.load(f)
        with open('park_mems.pkl', 'rb') as f:
            park_mems = pickle.load(f)
        print("Successfully loaded fuzzy distance matrices from .pkl files.")
    except FileNotFoundError:
        print("ERROR: Could not find 'interv_mems.pkl' or 'park_mems.pkl'.")
        print("Cannot compute closeness or environmental impact concurrency. Exiting.")
        exit()

    # 4. Compute all metrics
    print("\nComputing all concurrency metrics...")
    sol.compute_concurrency(instance)
    sol.compute_seansonality(instance)
    sol.compute_size_concurrency(instance)
    sol.compute_risk_concurrency(instance)
    sol.dist_matrix_to_closeness_concurrency(interv_mems)
    sol.compute_environmental_impact_concurrency(park_mems, tconorm=np.maximum)
    print("Computation complete.")

    # 5. Print the final entropy scores
    print("\n" + "="*40)
    print("      FINAL CONCURRENCY ENTROPY SCORES")
    print("="*40)
    
    def print_scores(title, scores_dict):
        print(f"\n--- {title} ---")
        if not scores_dict:
            print("  No scores calculated.")
            return
        for label, score in scores_dict.items():
            print(f"  {label:<15}: {score:.4f}")

    print_scores("Size Concurrency", sol.size_concurrency)
    print_scores("Risk Concurrency", sol.risk_concurrency)
    print_scores("Closeness Concurrency", sol.closeness_concurrency)
    print_scores("Environmental Impact Concurrency", sol.environmental_impact_concurrency)
    print("="*40)

    # 6. Plot the results
    print("\nGenerating concurrency plots...")
    sol.plot_all_concurrency_details()