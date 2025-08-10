import numpy as np
from baldric.sampler import EmbeddingFreespaceSampler
from baldric.collision.aabb_collision import AABBCollisionChecker, AABB
from baldric.collision.convex_collision import (
    ConvexPolygon2dCollisionChecker,
    ConvexPolygon2dSet,
    ConvexPolygon2d,
)
from baldric.spaces import VectorSpace, RigidBody2dSpace
from baldric.metrics import VectorNearest
from baldric.planners.prm import PRM, PRMPlan


def test_prm_in_r2():
    space = VectorSpace()
    checker = AABBCollisionChecker(
        space=space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = EmbeddingFreespaceSampler(
        np.array([0.0, 0.0]), np.array([100.0, 100.0]), checker
    )
    nearest = VectorNearest(space)
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


def robot():
    robot = [
        np.array(
            [
                [-9, 20],
                [-11, 20],
                [-11, -1],
                [-9, -1],
            ]
        ),
        np.array(
            [
                [9, 20],
                [9, -1],
                [11, -1],
                [11, 20],
            ]
        ),
        np.array(
            [
                [-10.5, 1],
                [-10.5, -1],
                [10.5, -1],
                [10.5, 1],
            ]
        ),
    ]
    return ConvexPolygon2dSet(polys=[ConvexPolygon2d(p) for p in robot])


def test_prm_rigidbody2d():
    bot = robot()
    space = RigidBody2dSpace.from_points(np.vstack([p.pts for p in bot.polys]))
    obs_hgt = 30.0
    obs_pts = np.array([[45.0, 0.0], [55.0, 0.0], [55.0, obs_hgt], [45.0, obs_hgt]])
    obs = [
        ConvexPolygon2dSet(
            polys=[
                ConvexPolygon2d(obs_pts),
            ]
        )
    ]
    checker = ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )
    sampler = EmbeddingFreespaceSampler(
        np.array([0.0, 0.0, -np.pi]), np.array([100.0, 100.0, np.pi]), checker
    )
    nearest = VectorNearest(space)
    planner = PRM(sampler, nearest, checker, r=100, n=20)
    planner.prepare()
    q_i = np.array([10.0, 10.0, 0.0])
    q_f = np.array([90.0, 10.0, 0.0])
    plan = planner.plan(q_i, q_f)
    assert plan is not None
    assert isinstance(plan, PRMPlan)
    assert isinstance(plan.path_indices, list)
