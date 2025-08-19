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

    def valid(self, q) -> bool:
        if len(q) != self._dimension:
            return False
        if np.any(q < self._low):
            return False
        if np.any(q > self._high):
            return False
        return True

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


class PiecewisePath:
    """A piecewise path is made up of paths between configurations"""

    def __init__(self, space: Space, qs: ConfigurationSet):
        self._space = space
        self._qs = qs

        # lengths and cummulative lengths
        lengths = np.zeros(self._qs.shape[0] - 1)
        cumlen = np.zeros(self._qs.shape[0])
        for i in range(self.nsegments):
            qi = self._qs[i]
            qj = self._qs[i + 1]
            lengths[i] = self._space.distance(qi, qj)
        cumlen[1:] = np.cumsum(lengths)
        self.segment_lengths = lengths
        self.cumlen = cumlen
        self._total_length = np.sum(self.segment_lengths)

    @property
    def configurations(self):
        return self._qs

    @property
    def nconfigurations(self):
        return self._qs.shape[0]

    @property
    def nsegments(self):
        return self._qs.shape[0] - 1

    @property
    def length(self):
        return self._total_length

    def interpolate(self, ss: np.ndarray) -> ConfigurationSet:
        """Interpolate the piecewise path

        ss : np.ndarray
            Scalars from 0..length
        """
        xs = np.searchsorted(self.cumlen, ss, side="right")
        xs = np.minimum(np.maximum(xs - 1, 0), self.nsegments - 1)
        delta = ss - self.cumlen[xs]
        gaps = self.segment_lengths[xs]
        ss = delta / gaps
        res = []
        for i in range(len(ss)):
            # bare interpolation falls back to the space interpolator
            q = self._space.interpolate(self._qs[xs[i]], self._qs[xs[i] + 1], ss[i])
            res.append(q)
        return np.vstack(res)

    def interpolate_with_step(self, path: ConfigurationSet, step: float) -> ConfigurationSet:
        """Interpolate the piecewise path with a minimum step size

        step : float
            value to traverse along the path
        """
        nsteps = np.ceil(self.length / step)
        nsteps = int(max(nsteps, 1)) + 1
        ss = np.linspace(0.0, self.length, nsteps)
        return self.interpolate(path, ss)
