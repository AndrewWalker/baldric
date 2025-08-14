from dataclasses import dataclass
import numpy as np
import networkx as nx
from loguru import logger
from baldric.core import FreespaceSampler, CollisionChecker, Nearest, Planner


@dataclass
class PRMPlan:
    g: nx.Graph
    path_indices: np.ndarray
    qs: np.ndarray

    @property
    def path(self):
        return np.vstack([self.qs[i] for i in self.path_indices])


class PlannerPRM(Planner[PRMPlan]):
    def __init__(
        self,
        sampler: FreespaceSampler,
        nearest: Nearest,
        colltest: CollisionChecker,
        n=1000,
        r=20.0,
    ):
        super().__init__(colltest)
        self.n = n
        self.r = r
        self.qs = None
        self.sampler = sampler
        self.nearest = nearest
        self._prepared = False

    def sample(self):
        qs = []
        for i in range(self.n):
            q = self.sampler.sampleFree()
            if q is not None:
                qs.append(q)
        self.qs = np.array(qs)
        return self.qs

    def connectOne(self, qs, i):
        es = []
        candidates = self.nearest.near(qs, qs[i, :], self.r)
        for j in candidates:
            j = int(j)
            if j != i:
                q_i = qs[i, :]
                q_j = qs[j, :]
                if self.collisionFreeSegment(q_i, q_j):
                    es.append((i, j))
        return es

    def connect(self):
        es = []
        n = self.qs.shape[0]
        for i in range(n):
            logger.info(f"connecting {i} of {n} #edges = {len(es)}")
            es += self.connectOne(self.qs, i)
        self.es = es
        return es

    def constructGraph(self, qs, es):
        g = nx.Graph()
        for i in range(len(self.qs)):
            g.add_node(i)
        for i, j in es:
            q_i = qs[i]
            q_j = qs[j]
            w = self.nearest.distance(q_i, q_j)
            g.add_edge(i, j, weight=w)
        return g

    def prepare(self):
        if not self._prepared:
            self.sample()
            self.connect()
            self._prepared = True

    def plan(self, q_i, q_f):
        if not self.collisionFree(q_i):
            logger.debug("initial configuration not collision free")
            return None
        if not self.collisionFree(q_f):
            logger.debug("goal configuration not collision free")
            return None

        self.prepare()
        src = len(self.qs)
        dst = len(self.qs) + 1
        qs = np.vstack([self.qs, q_i, q_f])
        es = list(self.es)
        es += self.connectOne(qs, src)
        es += self.connectOne(qs, dst)
        g = self.constructGraph(qs, es)
        try:
            path = nx.shortest_path(g, src, dst, "weight")
            return PRMPlan(g, path, qs)
        except nx.NetworkXNoPath:
            ncpts = len(list(nx.connected_components(g)))
            logger.info(f"no path, graph has {ncpts} components")
        return None
