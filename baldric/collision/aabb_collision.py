from typing import List
from baldric.collision import CollisionChecker
from baldric.spaces import VectorSpace
from baldric.geometry import AABB


class AABBCollisionChecker(CollisionChecker):
    def __init__(self, space: VectorSpace, boxes: List[AABB], robot: AABB, step: float):
        super().__init__(space)
        self.boxes = boxes
        self.maxStep = step
        self.robot = robot

    def collisionFree(self, x) -> bool:
        for aabb in self.boxes:
            if aabb.intersectsAABB(self.robot):
                return False
        return True

    def collisionFreeSegment(self, q0, q1):
        for q in self._space.interpolate_approx_distance(q0, q1, self.maxStep):
            if not self.collisionFree(q):
                return False
        return True
