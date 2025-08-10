import numpy as np
from baldric.sampler import EmbeddingFreespaceSampler
from baldric.collision.aabb_collision import AABBCollisionChecker, AABB
from baldric.spaces import VectorSpace
from baldric.metrics import VectorNearest
from baldric.planners.prm import PRM, PRMPlan


def test_prm_in_r2():
    space = VectorSpace(2)
    checker = AABBCollisionChecker(
        space=space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = EmbeddingFreespaceSampler(
        np.array([0.0, 0.0]), np.array([100.0, 100.0]), checker
    )
    nearest = VectorNearest()
    planner = PRM(sampler, nearest, checker, r=100, n=20)
    planner.prepare()
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    plan = planner.plan(q_i, q_f)
    assert plan is not None
    assert isinstance(plan, PRMPlan)
    assert isinstance(plan.path_indices, list)
    # we can calculate a rough analytic lower bound here
    # 2 * sqrt(2) * 40 ~ 112 units. this is enough to show
    # the planner didn't bypass the obstacles
    assert space.piecewise_path_length(plan.path) > 112.0
