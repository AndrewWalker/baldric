import numpy as np
from .space import Space, Configuration, ConfigurationSet
from . import dubins
from loguru import logger


class DubinsSpace(Space):
    """R^2 . S"""

    def __init__(self, low, high, rho=1.0):
        super().__init__(low, high)
        self.rho = rho

    def normalize_angle(self, angle):
        res = np.arctan2(np.sin(angle), np.cos(angle))
        return res

    def difference(self, q0: Configuration, q1: Configuration):
        dq = (q1 - q0).reshape(-1, self.dimension)
        dq[:, 2] = self.normalize_angle(dq[:, 2])
        return dq

    def normalise(self, q):
        res = q.copy()
        res[:, 2] = self.normalize_angle(q[:, 2])
        return res

    def distance(self, q0: Configuration, q1: Configuration) -> float:
        """Calculate the distance between configurations"""
        return dubins.shortest_path(q0, q1, self.rho).length

    def distance_many(self, qs: ConfigurationSet, q1: Configuration) -> np.ndarray:
        dists = []
        for q0 in qs:
            dists.append(self.distance(q0, q1))

    def interpolate(self, q0: Configuration, q1: Configuration, s: float) -> Configuration:
        """Interpolate between configurations"""
        dq = self.difference(q0, q1)
        return self.normalise(q0 + s * dq)

    def interpolate_many(self, q0: Configuration, q1: Configuration, ss: np.ndarray) -> ConfigurationSet:
        lst = []
        pth = dubins.shortest_path(q0, q1, self.rho)
        for i in range(ss.shape[0]):
            q = pth.i
            lst.append(q)
        return np.vstack(lst)
