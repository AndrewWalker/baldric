import numpy as np
from baldric.collision import CollisionChecker


class FreespaceSampler:
    def __init__(self, colltest: CollisionChecker):
        super().__init__()
        self._colltest = colltest

    def sample(self) -> np.ndarray:
        """Produce a sample in the configuration space"""
        raise NotImplementedError

    def sampleFree(self) -> np.ndarray | None:
        """Produce a collision-free sample in the configuration space"""
        x = self.sample()
        if self._colltest.collisionFree(x):
            return x
        return None


class EmbeddingFreespaceSampler(FreespaceSampler):
    """Uniform random sampling in a bounded configuration space"""

    def __init__(self, low: np.ndarray, high: np.ndarray, colltest: CollisionChecker):
        super().__init__(colltest)
        self._low = low
        self._high = high
        self._size = high - low
        self._dim = self._size.shape

    def sample(self) -> np.ndarray:
        return np.random.random(self._dim) * self._size + self._low
