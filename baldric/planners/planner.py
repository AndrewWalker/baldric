import numpy as np
from typing import List, Union, Callable
from baldric.collision import CollisionChecker
from baldric.spaces import Space, Configuration
from typing import Generic, TypeVar

PlanT = TypeVar("PlanT")


class GoalBase:
    def satisified(self, q):
        return False


class DiscreteGoal(GoalBase):
    def __init__(self, location: Configuration, tolerance: float, space: Space):
        super().__init__()
        self.location = location
        self.space = space
        self.tolerance = tolerance

    def satisified(self, q):
        return self.space.distance(self.location, q) < self.tolerance


class PredicateGoal(GoalBase):
    def __init__(self, location: Configuration, pred: Callable[[Configuration, Configuration], bool]):
        self.location = location
        self.pred = pred

    def satisified(self, q):
        return self.pred(self.location, q)


Goal = Union[DiscreteGoal, PredicateGoal]


class Planner(Generic[PlanT]):
    def __init__(self, colltest: CollisionChecker):
        self._colltest = colltest

    def collisionFree(self, x: Configuration) -> bool:
        return self._colltest.collisionFree(x)

    def collisionFreeSegment(self, q_0: Configuration, q_1: Configuration) -> bool:
        return self._colltest.collisionFreeSegment(q_0, q_1)

    def plan(self, x_init: Configuration, goal: Goal) -> PlanT:
        raise NotImplementedError
