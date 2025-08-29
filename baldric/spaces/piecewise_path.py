import numpy as np
from .space import Space
from baldric.configuration import ConfigurationSet


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

    def interpolate_with_step(self, step: float) -> ConfigurationSet:
        """Interpolate the piecewise path with a minimum step size

        step : float
            value to traverse along the path
        """
        nsteps = np.ceil(self.length / step)
        nsteps = int(max(nsteps, 1)) + 1
        ss = np.linspace(0.0, self.length, nsteps)
        return self.interpolate(ss)
