import numpy as np
from .space import Space, Configuration
from loguru import logger


class RigidBody2dSpace(Space):
    """R^2 . S"""

    def __init__(self, low, high):
        super().__init__(low, high)

    def set_weights(self, weights: np.ndarray):
        logger.info(f"setting weights {str(weights)}")
        self.metric_weights = weights

    def set_weights_from_pts(self, pts: np.ndarray):
        L = np.max(np.linalg.norm(pts, axis=1))
        logger.info(f"maximum lever arm is {L}")
        weights = np.array([1.0, 1.0, 1.45])
        self.set_weights(weights)

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
