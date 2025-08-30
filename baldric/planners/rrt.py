import dataclasses
import numpy as np
from typing import Iterator, Tuple
from loguru import logger
from baldric.spaces import Space, PiecewisePath
from baldric.sampler import FreespaceSampler
from baldric.collision import CollisionChecker
from baldric.metrics import Nearest
from .planner import Planner, Goal


@dataclasses.dataclass
class Tree:
    configuration: np.ndarray
    parent: np.ndarray
    n: int

    def __init__(self, maximumNodes=100, qdims=3):
        self.parent = np.zeros(maximumNodes, dtype=np.int32)
        self.configuration = np.zeros((maximumNodes, qdims))
        self.n = 0

    @property
    def activeConfigurations(self):
        return self.configuration[: self.n, :]

    @property
    def numNodes(self):
        return self.n

    @property
    def edges(self) -> Iterator[Tuple[int, int]]:
        return ((i, int(self.parent[i])) for i in range(1, self.n))

    def insert(self, q: np.ndarray, parent: int | None = None) -> int:
        if parent is None:
            parent = -1
        self.parent[self.n] = parent
        self.configuration[self.n, :] = q
        idx = self.n
        self.n += 1
        return idx


@dataclasses.dataclass
class RRTPlan:
    space: Space
    t: Tree
    soln_idx: int

    def path_indices(self):
        if self.soln_idx is None:
            return None
        soln = []
        idx: int | None = self.soln_idx
        while True:
            idx = int(self.t.parent[idx])
            if idx == -1:
                break
            soln.append(idx)
        soln.reverse()
        return soln

    @property
    def path(self):
        indices = self.path_indices()
        if indices is None:
            return None
        pth = np.vstack([self.t.configuration[i, :] for i in indices])
        return PiecewisePath(self.space, pth)


class PlannerRRT(Planner[RRTPlan]):
    def __init__(
        self,
        sampler: FreespaceSampler,
        colltest: CollisionChecker,
        nearest: Nearest,
        n: int = 500,
        eta: float = 1.0,
    ):
        super().__init__(colltest)
        assert eta > 0
        assert n > 0
        self._n = n
        self._eta = eta
        self._qdims = colltest._space.dimension
        self._sampler = sampler
        self._nearest = nearest

    @property
    def space(self):
        return self._colltest._space

    def steer(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dist = self.space.distance(x, y)
        dist = min(self._eta / dist, 1.0)
        return self.space.interpolate(x, y, dist)

    def nearest(self, qs, q):
        return self._nearest.nearest(qs, q)

    def freeSample(self):
        x_rand = None
        while x_rand is None:
            x_rand = self._sampler.sampleFree()
        return x_rand

    def plan(self, x_init: np.ndarray, ingoal: Goal) -> RRTPlan:
        logger.info("solving")
        assert self.collisionFree(x_init)
        tree = Tree(maximumNodes=self._n, qdims=self._qdims)
        tree.insert(x_init)
        soln_idx: int | None = None
        while tree.n < self._n:
            x_rand = self.freeSample()
            x_near_idx = self.nearest(tree.activeConfigurations, x_rand)
            x_near = tree.configuration[x_near_idx]
            x_steer = self.steer(x_near, x_rand)
            if self._colltest.collisionFreeSegment(x_near, x_steer):
                idx = tree.insert(x_steer, parent=x_near_idx)
                if ingoal.satisified(x_steer):
                    soln_idx = idx
                    logger.info(f"done early {tree.n}")
                    break
        if soln_idx is None:
            logger.info("no solution")
        logger.info("solving:done")
        return RRTPlan(self.space, t=tree, soln_idx=soln_idx)
