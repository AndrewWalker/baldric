import numpy as np


class Space:
    def __init__(self, d):
        self._dimension = d

    @property
    def dimension(self) -> int:
        return self._dimension

    def normalise_state(self, q: np.ndarray) -> np.ndarray:
        return q

    def distance(self, q0: np.ndarray, q1: np.ndarray) -> float:
        raise NotImplementedError

    def interpolate(self, q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
        raise NotImplementedError

    def interpolate_many(
        self, q0: np.ndarray, q1: np.ndarray, ss: np.ndarray
    ) -> np.ndarray:
        lst = []
        for i in range(ss.shape[0]):
            q = self.interpolate(q0, q1, ss[i])
            lst.append(q)
        return np.vstack(lst)

    def interpolate_linspace(
        self, q0: np.ndarray, q1: np.ndarray, n: int
    ) -> np.ndarray:
        ss = np.linspace(0.0, 1.0, n)
        return self.interpolate_many(q0, q1, ss)

    def _length_arange(self, length, step) -> np.ndarray:
        nsteps = np.ceil(length / step)
        nsteps = int(max(nsteps, 1)) + 1
        return np.linspace(0.0, length, nsteps)

    def interpolate_approx_distance(
        self, q0: np.ndarray, q1: np.ndarray, step: float
    ) -> np.ndarray:
        dist = self.distance(q0, q1)
        ss = self._length_arange(dist, step) / dist
        return self.interpolate_many(q0, q1, ss)

    def piecewise_path_length(self, path: np.ndarray) -> float:
        path_cnt = path.shape[0]
        diff_lengths = np.zeros(path_cnt)
        for i in range(path_cnt - 1):
            diff_lengths[i + 1] = self.distance(path[i, :], path[i + 1, :])
        return np.sum(diff_lengths)

    def interpolate_piecewise_path(
        self, path: np.ndarray, ss: np.ndarray
    ) -> np.ndarray:
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
        self, path: np.ndarray, step: float
    ) -> np.ndarray:
        path_cnt = path.shape[0]
        diff_lengths = np.zeros(path_cnt)
        for i in range(path_cnt - 1):
            diff_lengths[i + 1] = self.distance(path[i, :], path[i + 1, :])
        length = np.sum(diff_lengths)
        ss = self._length_arange(length, step)
        return self.interpolate_piecewise_path(path, ss)


class VectorSpace(Space):
    """R^n"""

    def __init__(self, d=2):
        super().__init__(d=d)

    def distance(self, q0: np.ndarray, q1: np.ndarray) -> float:
        return np.linalg.norm(q1 - q0)

    def interpolate(self, q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
        return q0 + s * (q1 - q0)


class RigidBody2dSpace(Space):
    """R^2 . S"""

    def __init__(self):
        super().__init__(d=2)

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def normalise_state(self, q: np.ndarray) -> np.ndarray:
        res = q.copy()
        res[2] = self.normalize_angle(q[2])
        return res

    def distance(self, q0: np.ndarray, q1: np.ndarray) -> float:
        return np.linalg.norm(q1[:2] - q0[:2])

    def interpolate(self, q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
        dq = self.normalise_state(q1 - q0)
        return q0 + s * dq
