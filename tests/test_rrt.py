import numpy as np
from baldric.sampler import FreespaceSampler
from baldric.collision import aabb_collision as box_cc
from baldric.metrics import VectorNearest
from baldric.planners import PlannerRRT, DiscreteGoal
from baldric.spaces import VectorSpace


def test_rrt_in_r2():
    space = VectorSpace(np.array([0.0, 0.0]), np.array([100.0, 100.0]))
    checker = box_cc.AABBCollisionChecker(
        space=space,
        boxes=[box_cc.AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerRRT(sampler, checker, nearest, n=20, eta=5.0)
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    planner.plan(q_i, DiscreteGoal(q_f, 5.0, space))
