from dataclasses import dataclass
from typing import TypeVar, Generic
from baldric.spaces import Space, Configuration
from baldric.sampler import FreespaceSampler
from baldric.planners import Planner, Goal
from baldric.collision import CollisionChecker
from baldric.metrics import Nearest


PlanT = TypeVar("PlanT")


@dataclass
class Problem(Generic[PlanT]):
    init = Configuration
    goal = Goal
    space: Space | None = None
    nearest: Nearest | None = None
    collision_checker: CollisionChecker | None = None
    sampler: FreespaceSampler | None = None
    planner: Planner[PlanT] | None = None
