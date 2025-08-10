import numpy as np
from baldric.collision import CollisionChecker
from typing import Generic, TypeVar

PlanT = TypeVar("PlanT")


class Planner(Generic[PlanT]):
    def __init__(self, colltest: CollisionChecker):
        self._colltest = colltest

    def collisionFree(self, x: np.ndarray) -> bool:
        return self._colltest.collisionFree(x)

    def collisionFreeSegment(self, q_0: np.ndarray, q_1: np.ndarray) -> bool:
        return self._colltest.collisionFreeSegment(q_0, q_1)

    def plan(self, x_init: np.ndarray) -> PlanT | None:
        return None
