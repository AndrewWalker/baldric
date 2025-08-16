import numpy as np
from baldric.sampler import FreespaceSampler
from baldric.collision import aabb_collision as box_cc
from baldric.collision import convex_collision as cvx_cc
from baldric.spaces import VectorSpace, RigidBody2dSpace
from baldric.metrics import VectorNearest
from baldric.planners import PlannerPRM, PRMPlan, DiscreteGoal


def test_prm_in_r2():
    space = VectorSpace(np.array([0.0, 0.0]), np.array([100.0, 100.0]))
    checker = box_cc.AABBCollisionChecker(
        space=space,
        boxes=[box_cc.AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerPRM(sampler, nearest, checker, r=100, n=20)
    planner.prepare()
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    plan = planner.plan(q_i, DiscreteGoal(location=q_f, tolerance=1.0, space=space))
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
    return cvx_cc.ConvexPolygon2dSet(polys=[cvx_cc.ConvexPolygon2d(p) for p in robot])


def test_prm_rigidbody2d():
    bot = robot()

    obs_hgt = 30.0
    obs_pts = np.array([[45.0, 0.0], [55.0, 0.0], [55.0, obs_hgt], [45.0, obs_hgt]])
    obs = cvx_cc.ConvexPolygon2dSet(
        polys=[
            cvx_cc.ConvexPolygon2d(obs_pts),
        ]
    )

    space = RigidBody2dSpace(
        np.array([0.0, 0.0, -np.pi]), np.array([100.0, 100.0, np.pi])
    )
    space.set_weights_from_pts(np.vstack([p.pts for p in bot.polys]))

    checker = cvx_cc.ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )
    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerPRM(sampler, nearest, checker, r=100, n=20)
    planner.prepare()
    q_i = np.array([10.0, 10.0, 0.0])
    q_f = np.array([90.0, 10.0, 0.0])
    plan = planner.plan(q_i, DiscreteGoal(location=q_f, tolerance=1.0, space=space))
    assert plan is not None
    assert isinstance(plan, PRMPlan)
    assert isinstance(plan.path_indices, list)
