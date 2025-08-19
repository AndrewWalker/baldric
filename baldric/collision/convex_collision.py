from typing import List
import numpy as np
from baldric.spaces import RigidBody2dSpace
from baldric.collision import CollisionChecker
from .gjk import gjk


def rotation_matrix_2d(theta: float):
    cos_angle = np.cos(theta)
    sin_angle = np.sin(theta)
    A = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return A


class ConvexPolygon2d:
    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.mu = np.mean(self.pts, axis=0)
        self.r = np.max(np.linalg.norm(self.pts - self.mu, axis=0))

    @staticmethod
    def _rotate(pts, theta):
        return pts @ rotation_matrix_2d(theta)

    @staticmethod
    def _translate(pts, offset):
        return pts + offset

    def transform(self, x, y, theta):
        pts = self.pts
        pts = ConvexPolygon2d._rotate(pts, theta)
        pts = ConvexPolygon2d._translate(pts, np.array([x, y]))
        return ConvexPolygon2d(pts)

    def mean(self):
        return self.mu

    def radius(self):
        return self.r


class ConvexPolygon2dSet:
    def __init__(self, polys: List[ConvexPolygon2d]):
        self.polys = polys

    @property
    def all_points(self):
        return np.vstack([p.pts for p in self.polys])

    def transform(self, q):
        res = []
        for p in self.polys:
            pt = p.transform(q[0], q[1], q[2])
            res.append(pt)
        return ConvexPolygon2dSet(res)


class ConvexPolygon2dCollisionChecker(CollisionChecker):
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

    def collisionFree(self, q: np.ndarray) -> bool:
        assert self.space.valid(q)
        for p in self.obs.polys:
            trobot = self.robot.transform(q).polys
            for rpoly in trobot:
                if gjk(p.pts, rpoly.pts):
                    return False
        return True

    def collisionFreeSegment(self, q0: np.ndarray, q1: np.ndarray) -> bool:
        pts = self.space.interpolate_approx_distance(q0, q1, self.maxStep)
        for q in pts:
            if not self.collisionFree(q):
                return False
        return True
