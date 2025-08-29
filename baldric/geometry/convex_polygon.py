import numpy as np
from typing import List


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
