from dataclasses import dataclass
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


@dataclass
class Problem(Generic[PlanT]):
    init = Configuration
    goal = Goal
    space: Space | None = None
    nearest: Nearest | None = None
    collision_checker: CollisionChecker | None = None
    sampler: FreespaceSampler | None = None
    planner: Planner[PlanT] | None = None
