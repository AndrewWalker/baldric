from dataclasses import dataclass
import numpy as np
import networkx as nx
from loguru import logger
from baldric.sampler import FreespaceSampler
from baldric.collision import CollisionChecker
from baldric.metrics import Nearest
from baldric.planners import Planner, Goal, DiscreteGoal
from baldric.spaces import Configuration
import math


def unit_ball_measure(dim: int):
    return math.sqrt(math.pi) ** dim / math.gamma(dim / 2 + 1)


def neighbour_radius(space_measure: float, dim: int, num_states: int):
    inverse_dim = 1.0 / dim
    space_measure_ratio = space_measure / unit_ball_measure(dim)
    prm_constant = (
        2 * (1 + inverse_dim) ** inverse_dim * space_measure_ratio**inverse_dim
    )
    gamma_scale = 2.0
    return (
        gamma_scale * prm_constant * (math.log(num_states) / num_states) ** inverse_dim
    )


@dataclass
class PRMPlan:
    g: nx.Graph
    qs: np.ndarray
    es: np.ndarray
    path_indices: np.ndarray | None = None

    @property
    def path(self):
        return np.vstack([self.qs[i] for i in self.path_indices])


class PlannerPRM(Planner[PRMPlan]):
    def __init__(
        self,
        sampler: FreespaceSampler,
        nearest: Nearest,
        colltest: CollisionChecker,
        n: int = 1000,
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
        while len(qs) < self.n:
            q = self.sampler.sampleFree()
            if q is not None:
                qs.append(q)
        self.qs = np.array(qs)
        return self.qs

    def connectOne(self, es, qs, i):
        candidates = self.nearest.near(qs, qs[i, :], self.r)
        for j in candidates:
            j = int(j)
            if j == i:
                continue
            if (j, i) in es:
                continue
            q_i = qs[i, :]
            q_j = qs[j, :]
            if self.collisionFreeSegment(q_i, q_j):
                es.append((i, j))

    def connect(self):
        es = []
        n = self.qs.shape[0]
        for i in range(n):
            # logger.info(f"connecting {i} of {n} #edges = {len(es)}")
            self.connectOne(es, self.qs, i)
        self.es = es
        return es

    def constructGraph(self, qs, es):
        g = nx.Graph()
        for i in range(len(qs)):
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

    def query(self, q_i: Configuration, q_f: Configuration):
        src = len(self.qs)
        dst = len(self.qs) + 1
        qs = np.vstack([self.qs, q_i, q_f])
        es = list(self.es)
        self.connectOne(es, qs, src)
        self.connectOne(es, qs, dst)
        g = self.constructGraph(qs, es)
        path = None
        try:
            path = nx.shortest_path(g, src, dst, "weight")
        except nx.NetworkXNoPath:
            ncpts = len(list(nx.connected_components(g)))
            logger.info(f"no path, graph has {ncpts} components")
        return PRMPlan(g=g, qs=qs, es=es, path_indices=path)

    def resolve_goal(self, goal: Goal):
        match goal:
            case DiscreteGoal():
                return goal.location
            case _:
                raise RuntimeError(f"Invalid goal type {type(goal)} for PRM")

    def plan(self, q_i: Configuration, goal: Goal) -> PRMPlan | None:
        q_f = self.resolve_goal(goal)
        if not self.collisionFree(q_i):
            logger.debug("initial configuration not collision free")
            return None
        if not self.collisionFree(q_f):
            logger.debug("goal configuration not collision free")
            return None

        logger.info(f"ideal r is {neighbour_radius(100, 2, self.n)}")
        self.prepare()
        return self.query(q_i, q_f)
