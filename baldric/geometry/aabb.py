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
        # required that the limits are positive
        assert np.all(qlim > 0)
        self._qcen = qcen
        self._qlim = qlim

    def transform(self, v: np.ndarray):
        return AABB(self.center + v, self.limits)

    def containsPt(self, q: np.ndarray):
        """Check if this AABB contains a point

        Parameters
        ----------
        q : np.ndarray
            The test point
        """
        return np.all(np.abs(q - self._qcen) < self._qlim)

    def intersectsAABB(self, other: "AABB"):
        """Check if this AABB intersects another AABB

        Parameters
        ----------
        other : AABB
            The other AABB
        """
        return AABB(self.center, self.limits + other.limits).containsPt(other.center)

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
