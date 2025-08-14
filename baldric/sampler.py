import numpy as np
from baldric.collision import CollisionChecker


class FreespaceSampler:
    def __init__(self, checker: CollisionChecker):
        super().__init__()
        self._checker = checker
        self._size = self.space._high - self.space._low

    @property
    def space(self):
        return self._checker.space

    def sample(self) -> np.ndarray:
        """Produce a sample in the configuration space"""
        return np.random.random(self.space.dimension) * self._size + self.space._low

    def sampleFree(self) -> np.ndarray | None:
        """Produce a collision-free sample in the configuration space"""
        x = self.sample()
        if self._checker.collisionFree(x):
            return x
        return None
