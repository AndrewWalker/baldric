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
from baldric.planners.rrt import PlannerRRT


def test_rrt_in_r2():
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
    planner = PlannerRRT(sampler, checker, nearest, n=20, eta=5.0)
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    tree = planner.plan(q_i)
