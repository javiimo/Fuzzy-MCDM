import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any

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
    """
    start_time: int
    duration: int
    workloads: Dict[str, List[float]]
    risks: List[List[float]]

    def __str__(self) -> str:
        # Create a summary string for each resource's workload.
        if self.workloads:
            workloads_str = "\n    ".join(
                f"{res}: {vals} ({compute_stats(vals)})" 
                for res, vals in self.workloads.items()
            )
        else:
            workloads_str = "None"
        # Create a summary string for the risk profiles per timestep.
        if self.risks:
            risks_str = "\n    ".join(
                f"Relative period {i+1}: {r} ({compute_stats(r)})" 
                for i, r in enumerate(self.risks)
            )
        else:
            risks_str = "None"
        return (f"Schedule Option (start_time={self.start_time}, duration={self.duration}):\n"
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
    """
    name: str
    tmax: int
    options: List[ScheduleOption] = field(default_factory=list)
    
    def add_option(self, option: ScheduleOption):
        if option.start_time > self.tmax:
            raise ValueError(
                f"Option start time {option.start_time} exceeds tmax ({self.tmax}) for intervention {self.name}"
            )
        self.options.append(option)

    def __str__(self) -> str:
        if self.options:
            options_str = "\n  ".join(str(opt) for opt in self.options)
        else:
            options_str = "None"
        return (f"Intervention '{self.name}': tmax = {self.tmax}, {len(self.options)} option(s):\n"
                f"  {options_str}")

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
        print(f"Error reading instance horizon or scenarios: {e}")
        T = 0
        scenarios_number = []
    instance = MaintenanceSchedulingInstance(T=T, scenarios_number=scenarios_number)
    
    # Load Resources.
    for res_name, res_data in json_data.get("Resources", {}).items():
        try:
            resource = Resource(
                name=res_name,
                capacity_min=res_data["min"],
                capacity_max=res_data["max"]
            )
            instance.add_resource(resource)
        except Exception as e:
            print(f"Error loading resource '{res_name}': {e}")
            continue
    
    # Load Seasons.
    for season_name, periods in json_data.get("Seasons", {}).items():
        try:
            season = Season(name=season_name, periods=periods)
            instance.add_season(season)
        except Exception as e:
            print(f"Error loading season '{season_name}': {e}")
            continue
    
    # Load Interventions.
    for intv_name, intv_data in json_data.get("Interventions", {}).items():
        try:
            tmax = intv_data["tmax"]
        except Exception as e:
            print(f"Error loading tmax for intervention '{intv_name}': {e}")
            continue
        try:
            intervention = Intervention(name=intv_name, tmax=tmax)
        except Exception as e:
            print(f"Error creating intervention '{intv_name}': {e}")
            continue
        
        delta_list = intv_data.get("Delta", [])
        workload_data = intv_data.get("workload", {})
        risk_data = intv_data.get("risk", {})
        
        # Iterate over each possible start time option (using 1-indexing).
        for i, duration in enumerate(delta_list, start=1):
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
                print(f"Error loading workload for intervention '{intv_name}', option {i}: {e}")
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
                print(f"Error loading risk for intervention '{intv_name}', option {i}: {e}")
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
                print(f"Error adding schedule option for intervention '{intv_name}', option {i}: {e}")
                continue
        
        try:
            instance.add_intervention(intervention)
        except Exception as e:
            print(f"Error adding intervention '{intv_name}': {e}")
            continue
    
    # Load Exclusions.
    for excl_key, excl_list in json_data.get("Exclusions", {}).items():
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
            print(f"Error loading exclusion '{excl_key}': {e}")
            continue
    
    return instance

    return instance

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    #Load Json
    json_path = 'challenge-roadef-2020/example1.json'
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
    instance = load_instance_from_json(example_json)
    
    # Print the instance with detailed metrics.
    #print(instance)
    instance.show()
