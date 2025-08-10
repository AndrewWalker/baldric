import numpy as np
from typing import List
from baldric.collision import CollisionChecker
from baldric.spaces import VectorSpace


class AABB:
    def __init__(self, qcen, qlim):
        self._qcen = qcen
        self._qlim = qlim

    def containsPt(self, q):
        return np.all(np.abs(q - self._qcen) < self._qlim)


class AABBCollisionChecker(CollisionChecker):
    def __init__(self, space: VectorSpace, boxes: List[AABB], step: float):
        super().__init__(space)
        self.boxes = boxes
        self.maxStep = step

    def collisionFree(self, x) -> bool:
        for aabb in self.boxes:
            if aabb.containsPt(x):
                return False
        return True

    def collisionFreeSegment(self, q0, q1):
        for q in self._space.interpolate_approx_distance(q0, q1, self.maxStep):
            if not self.collisionFree(q):
                return False
        return True
