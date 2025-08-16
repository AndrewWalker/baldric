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


class CachingCollisionChecker(CollisionChecker):
    def __init__(self, other: CollisionChecker):
        self._other = other
        self._locs = []

    @property
    def space(self):
        return self._other.space

    def collisionFree(self, x: np.ndarray) -> bool:
        """Check if a single configuration is collision free"""
        self._locs.append(x)
        return self._other.collisionFree(x)

    def collisionFreeSegment(self, q0: np.ndarray, q1: np.ndarray) -> bool:
        """Check if a path between configurations is collision free"""
        return self._other.collisionFreeSegment(q0, q1)
