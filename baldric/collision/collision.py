import numpy as np
from baldric.spaces import Space


class CollisionChecker:
    """Interface to check collisions"""

    def __init__(self, space: Space):
        self._space = space

    @property
    def space(self):
        return self._space

    def collisionFree(self, x: np.ndarray) -> bool:
        """Check if a single configuration is collision free"""
        return True

    def collisionFreeSegment(self, q_0: np.ndarray, q_1: np.ndarray) -> bool:
        """Check if a path between configurations is collision free"""
        return True
