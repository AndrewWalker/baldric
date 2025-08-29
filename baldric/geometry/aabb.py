import numpy as np


class AABB:
    """A N-dimensional axis aligned bounding box"""

    def __init__(self, qcen: np.ndarray, qlim: np.ndarray):
        """Create an axis aligned bounding box

        Parameters
        ----------
        qcen
            The center of the box
        qlim
            The maximum absolute distance from the center any edge
        """
        self._qcen = qcen
        self._qlim = qlim

    def containsPt(self, q: np.ndarray):
        """Check if a point is contained in the box

        Parameters
        ----------
        q : np.ndarray
            The test point
        """
        return np.all(np.abs(q - self._qcen) < self._qlim)

    @property
    def center(self):
        return self._qcen

    @property
    def limits(self):
        return self._qlim

    def sample(self):
        """Generate a uniform sample in the box"""
        s = np.random.random(self._qlim.shape) * 2 - 1
        return s * self._qlim + self._qcen
