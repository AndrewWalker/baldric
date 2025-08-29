from numpy.typing import NDArray, ArrayLike
import numpy as np
from loguru import logger
from baldric.configuration import Configuration, ConfigurationSet


class Space:
    def __init__(
        self,
        low: ArrayLike,
        high: ArrayLike,
    ):
        self._low = np.asarray(low)
        self._high = np.asarray(high)

        # sizes must match
        assert len(low) == len(high)
        self._dimension = len(low)

        # default the weights to something plausible
        self._weights = np.ones(self._dimension)

    def valid(self, q) -> bool:
        res = True
        if len(q) != self._dimension:
            res = False
        if np.any(q < self._low):
            res = False
        if np.any(q > self._high):
            res = False
        return res

    @property
    def dimension(self) -> int:
        return self._dimension

    def difference(self, q0: Configuration, q1: Configuration):
        """Find the difference between configurations as a vector

        Required to support a range of topological spaces
        """
        return q1 - q0

    def distance_many(self, qs: ConfigurationSet, q1: Configuration) -> np.ndarray:
        dq = self.difference(qs, q1)
        return np.sqrt(np.sum(dq * dq * self._weights, axis=1))

    def distance(self, q0: Configuration, q1: Configuration) -> float:
        """Calculate the distance between configurations"""
        return float(self.distance_many(q0.reshape(1, -1), q1)[0])

    def normalise(self, q: Configuration) -> Configuration:
        return q

    def interpolate(self, q0: Configuration, q1: Configuration, s: float) -> Configuration:
        """Interpolate between configurations"""
        if s < 0.0 or s > 1.0:
            snew = max(min(s, 1.0), 0.0)
            logger.debug(f"clamped interpolant {s} to {snew}")
            s = snew
        dq = self.difference(q0, q1)
        return self.normalise(q0 + s * dq)

    def interpolate_many(self, q0: Configuration, q1: Configuration, ss: np.ndarray) -> ConfigurationSet:
        lst = []
        for i in range(ss.shape[0]):
            q = self.interpolate(q0, q1, ss[i])
            lst.append(q)
        return np.vstack(lst)

    def interpolate_approx_distance(self, q0: Configuration, q1: Configuration, step: float) -> ConfigurationSet:
        # required for collision detection
        dist = self.distance(q0, q1)
        nsteps = np.ceil(dist / step)
        nsteps = int(max(nsteps, 1)) + 1
        ss = np.linspace(0.0, 1.0, nsteps)
        return self.interpolate_many(q0, q1, ss)
