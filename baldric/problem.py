from typing import TypeVar, Generic
from baldric.core import (
    Space,
    CollisionChecker,
    Planner,
    Goal,
    FreespaceSampler,
    Configuration,
    Nearest,
)


PlanT = TypeVar("PlanT")


class Problem(Generic[PlanT]):
    init = Configuration
    goal = Goal
    space: Space | None = None
    nearest: Nearest | None
    collision_checker: CollisionChecker | None = None
    sampler: FreespaceSampler
    planner: Planner[PlanT] | None = None
