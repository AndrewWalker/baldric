import numpy as np
from .space import Space, Configuration


class RigidBody2dSpace(Space):
    """R^2 . S"""

    def __init__(self, low, high):
        super().__init__(low, high)

    def set_weights(self, weights: np.ndarray):
        self.metric_weights = weights

    def set_weights_from_pts(self, pts: np.ndarray):
        L = np.max(np.linalg.norm(pts, axis=1))
        weights = np.array([1.0, 1.0, 2 * 1.45 * L])
        self.set_weights(weights)

    @staticmethod
    def from_points(pts: np.ndarray):
        space = RigidBody2dSpace()
        space.set_weights_from_pts(pts)
        return space

    def normalize_angle(self, angle):
        res = np.arctan2(np.sin(angle), np.cos(angle))
        return res

    def difference(self, q0: Configuration, q1: Configuration):
        dq = (q1 - q0).reshape(-1, self.dimension)
        dq[:, 2] = self.normalize_angle(dq[:, 2])
        return dq
