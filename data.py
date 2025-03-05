from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Resource:
    """
    Represents a resource available for maintenance work.
    
    Attributes:
        name: Unique identifier for the resource.
        capacity_min: List of minimum available capacity for each time period (length T).
        capacity_max: List of maximum available capacity for each time period (length T).
    """
    name: str
    capacity_min: List[float]
    capacity_max: List[float]


@dataclass
class Season:
    """
    Represents a season grouping certain time periods.
    
    Attributes:
        name: Name of the season (e.g., "winter", "summer", "full", "is").
        periods: List of time period indices belonging to this season.
    """
    name: str
    periods: List[int]


@dataclass
class Exclusion:
    """
    Represents a mutual exclusion constraint.
    
    Attributes:
        interventions: List of intervention names that must not overlap 
                       in the specified season.
        season: Name of the season during which the exclusions apply.
    """
    interventions: List[str]
    season: str


@dataclass
class Intervention:
    """
    Represents an intervention (maintenance task) to be scheduled.
    
    Attributes:
        name: Unique identifier for the intervention.
        tmax: Latest period at which the intervention may start so that it finishes on time.
        durations: Mapping from potential start time (1-indexed) to the intervention's duration.
                   For example, if the original "Delta" was [3, 3, 2] then you might store:
                   {1: 3, 2: 3, 3: 2}
        workloads: Mapping from resource name to another dictionary that maps each potential 
                   start time to its workload profile. The workload profile is a list of resource
                   consumption values for each relative time period of the intervention.
                   
                   Example:
                   {
                     "c1": {
                         1: [14, 0, 0],  # if started at time 1, resource c1 is used 14 in period 1, then 0...
                         2: [0, 14, 0],
                         3: [0, 0, 14]
                     }
                   }
        risks: Mapping from potential start time to a risk profile.
               The risk profile is a list (over the intervention's relative time steps) where 
               each element is itself a list of risk values—one per scenario—for that period.
               
               Example:
               {
                 1: [ [4, 8, 2], [0, 0], [0, 0, 0] ],  # If started at time 1, then:
                      # relative period 1: risks for 3 scenarios,
                      # relative period 2: risks for 2 scenarios,
                      # relative period 3: risks for 3 scenarios.
                 2: [ [0, 0, 0], [3, 8], [0, 0, 0] ]
               }
    """
    name: str
    tmax: int
    durations: Dict[int, int]  # key: possible start time, value: duration
    workloads: Dict[str, Dict[int, List[float]]]  # resource -> (start time -> workload profile)
    risks: Dict[int, List[List[float]]]  # start time -> list (over relative periods) of risk lists


@dataclass
class MaintenanceSchedulingInstance:
    """
    Represents a complete instance of the maintenance scheduling problem.
    
    Attributes:
        T: The number of time periods in the planning horizon.
        scenarios_number: A list giving the number of risk scenarios for each time period.
        resources: Dictionary of Resource objects keyed by resource name.
        seasons: Dictionary of Season objects keyed by season name.
        interventions: Dictionary of Intervention objects keyed by intervention name.
        exclusions: List of Exclusion constraints.
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

# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    # Define resources.
    r1 = Resource(name="c1", capacity_min=[10.0, 0.0, 6.0], capacity_max=[49.0, 23.0, 15.0])
    
    # Define seasons.
    season_winter = Season(name="winter", periods=[1, 2])
    season_summer = Season(name="summer", periods=[3])
    season_is = Season(name="is", periods=[])
    
    # Define an intervention.
    # For example, an intervention "I1" with tmax = 1 and possible durations [3, 3, 2]
    durations = {1: 3, 2: 3, 3: 2}
    
    # Workloads: for resource "c1", a mapping from potential start time to workload profile.
    workloads = {
        "c1": {
            1: [14, 0, 0],
            2: [0, 14, 0],
            3: [0, 0, 14]
        }
    }
    
    # Risks: mapping from start time to a risk profile.
    risks = {
        1: [ [4, 8, 2], [0, 0], [0, 0, 0] ],
        2: [ [0, 0, 0], [3, 8], [0, 0, 0] ]
        # (Note: start time 3 might be omitted if tmax limits the choices)
    }
    intervention_I1 = Intervention(name="I1", tmax=1, durations=durations, workloads=workloads, risks=risks)
    
    # Define an exclusion constraint.
    exclusion_E1 = Exclusion(interventions=["I2", "I3"], season="full")
    
    # Build the complete instance.
    instance = MaintenanceSchedulingInstance(
        T=3,
        scenarios_number=[3, 2, 3]
    )
    
    instance.add_resource(r1)
    instance.add_season(season_winter)
    instance.add_season(season_summer)
    instance.add_season(Season(name="full", periods=list(range(1, instance.T+1))))
    instance.add_season(season_is)
    instance.add_intervention(intervention_I1)
    instance.add_exclusion(exclusion_E1)
    
    # At this point, the instance object holds all the information in a clear and extendable structure.
    print(instance)
