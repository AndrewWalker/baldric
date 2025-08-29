import numpy as np
from baldric.spaces import RigidBody2dSpace
from baldric.collision import CollisionChecker
from baldric.geometry.gjk import gjk
from baldric.geometry.convex_polygon import ConvexPolygon2dSet, ConvexPolygon2d


class ConvexPolygon2dCollisionChecker(CollisionChecker):
    """Collision checker for two dimension convex polygons"""

    def __init__(
        self,
        space: RigidBody2dSpace,
        obs: ConvexPolygon2dSet,
        robot: ConvexPolygon2dSet,
        step: float = 1.0,
    ):
        super().__init__(space)
        self.obs = obs
        self.robot = robot
        self.maxStep = step
        self.checks = 0

    def collisionFree(self, q: np.ndarray) -> bool:
        if not self.space.valid(q):
            return False
        for p in self.obs.polys:
            trobot = self.robot.transform(q).polys
            for rpoly in trobot:
                self.checks += 1
                if gjk(p.pts, rpoly.pts):
                    return False
        return True

    def collisionFreeSegment(self, q0: np.ndarray, q1: np.ndarray) -> bool:
        pts = self.space.interpolate_approx_distance(q0, q1, self.maxStep)
        for q in pts:
            if not self.collisionFree(q):
                return False
        return True
