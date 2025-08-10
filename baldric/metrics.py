import numpy as np


class Nearest:
    def distance(self, qa, qb):
        return np.linalg.norm(qb - qa)

    def nearest(self, qs: np.ndarray, q: np.ndarray) -> int:
        """Find the index of the nearest element in qs to q"""
        raise NotImplementedError

    def near(self, qs: np.ndarray, q: np.ndarray, r: float) -> np.ndarray:
        """Find the indices of elements in qs within some distance r around q"""
        raise NotImplementedError


class NaiveNearest(Nearest):
    def __init__(self):
        super().__init__()

    def nearest(self, qs: np.ndarray, q: np.ndarray) -> int:
        assert qs.shape[1] == q.shape[0]
        best_i = None
        best_m = None
        for i in range(qs.shape[0]):
            m = self.distance(qs[i, :], q)
            if (best_m is None) or (m < best_m):
                best_m = m
                best_i = i
        return best_i

    def near(self, qs: np.ndarray, q: np.ndarray, r: float) -> np.ndarray:
        assert qs.shape[1] == q.shape[0]
        idxs = []
        for i in range(qs.shape[0]):
            m = self.distance(qs[i, :], q)
            if m < r:
                idxs.append(i)
        return np.array(idxs)


class VectorNearest(Nearest):
    def __init__(self):
        super().__init__()

    def nearest(self, qs: np.ndarray, c: np.ndarray) -> int:
        inp = qs - c
        tmp = np.linalg.norm(inp, axis=1)
        return np.argmin(tmp)

    def near(self, qs: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
        inp = qs - c
        norms = np.linalg.norm(inp, axis=1)
        assert norms.shape[0] == qs.shape[0]
        valid_mask = norms < r
        return np.where(valid_mask)[0]
