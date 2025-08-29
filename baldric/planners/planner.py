from baldric.collision import CollisionChecker
from baldric.configuration import Configuration
from baldric.goals import Goal
from typing import Generic, TypeVar

PlanT = TypeVar("PlanT")


class Planner(Generic[PlanT]):
    def __init__(self, colltest: CollisionChecker):
        self._colltest = colltest

    def collisionFree(self, x: Configuration) -> bool:
        return self._colltest.collisionFree(x)

    def collisionFreeSegment(self, q_0: Configuration, q_1: Configuration) -> bool:
        return self._colltest.collisionFreeSegment(q_0, q_1)

    def plan(self, x_init: Configuration, goal: Goal) -> PlanT:
        raise NotImplementedError
