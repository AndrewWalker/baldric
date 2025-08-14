from numpy.typing import NDArray, ArrayLike
import numpy as np
from loguru import logger

Configuration = NDArray[np.float64]
ConfigurationSet = NDArray[np.float64]


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

    def interpolate(
        self, q0: Configuration, q1: Configuration, s: float
    ) -> Configuration:
        """Interpolate between configurations"""
        if s < 0.0 or s > 1.0:
            snew = max(min(s, 1.0), 0.0)
            logger.debug("clamped interpolant {s} to {snew}")
            s = snew
        dq = self.difference(q0, q1)
        return q0 + s * dq

    def interpolate_many(
        self, q0: Configuration, q1: Configuration, ss: np.ndarray
    ) -> ConfigurationSet:
        lst = []
        for i in range(ss.shape[0]):
            q = self.interpolate(q0, q1, ss[i])
            lst.append(q)
        return np.vstack(lst)

    def interpolate_linspace(
        self, q0: Configuration, q1: Configuration, n: int
    ) -> np.ndarray:
        ss = np.linspace(0.0, 1.0, n)
        return self.interpolate_many(q0, q1, ss)

    def _length_arange(self, length, step) -> np.ndarray:
        nsteps = np.ceil(length / step)
        nsteps = int(max(nsteps, 1)) + 1
        return np.linspace(0.0, length, nsteps)

    def interpolate_approx_distance(
        self, q0: Configuration, q1: Configuration, step: float
    ) -> ConfigurationSet:
        dist = self.distance(q0, q1)
        ss = self._length_arange(dist, step) / dist
        return self.interpolate_many(q0, q1, ss)

    def piecewise_path_length(self, path: ConfigurationSet) -> float:
        path_cnt = path.shape[0]
        diff_lengths = np.zeros(path_cnt)
        for i in range(path_cnt - 1):
            diff_lengths[i + 1] = self.distance(path[i, :], path[i + 1, :])
        return np.sum(diff_lengths)

    def interpolate_piecewise_path(
        self, path: ConfigurationSet, ss: np.ndarray
    ) -> ConfigurationSet:
        path_cnt = path.shape[0]
        diff_lengths = np.zeros(path_cnt)
        for i in range(path_cnt - 1):
            diff_lengths[i + 1] = self.distance(path[i, :], path[i + 1, :])
        cumlen = np.cumsum(diff_lengths)
        xs = np.searchsorted(cumlen, ss, side="right")
        xs = np.minimum(np.maximum(xs - 1, 0), path_cnt - 2)
        delta = ss - cumlen[xs]
        gaps = diff_lengths[xs + 1]
        ss = delta / gaps
        res = []
        for i in range(len(ss)):
            q = self.interpolate(path[xs[i]], path[xs[i] + 1], ss[i])
            res.append(q)
        return np.vstack(res)

    def interpolate_piecewise_path_with_step(
        self, path: ConfigurationSet, step: float
    ) -> ConfigurationSet:
        path_cnt = path.shape[0]
        diff_lengths = np.zeros(path_cnt)
        for i in range(path_cnt - 1):
            diff_lengths[i + 1] = self.distance(path[i, :], path[i + 1, :])
        length = np.sum(diff_lengths)
        ss = self._length_arange(length, step)
        return self.interpolate_piecewise_path(path, ss)
