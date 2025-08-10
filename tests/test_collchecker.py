import numpy as np
from baldric.collision import CollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.collision.aabb_collision import (
    AABB,
    AABBCollisionChecker,
)
from baldric.spaces import RigidBody2dSpace, VectorSpace


def test_trivial_checker():
    checker = CollisionChecker(VectorSpace(3))


def test_aabb_checking():
    space = VectorSpace(d=2)
    o = AABB(np.array([0, 0.0]), np.array([1.0, 1.0]))
    checker = AABBCollisionChecker(space, [o], step=1.0)
    assert not checker.collisionFree(np.array([0, 0.0]))
    assert not checker.collisionFree(np.array([0.95, 0.0]))
    assert not checker.collisionFree(np.array([0.0, 0.95]))
    assert checker.collisionFree(np.array([1.05, 0.0]))
    assert checker.collisionFree(np.array([0.0, 1.05]))


def test_convex_checking():
    space = RigidBody2dSpace()
    r = ConvexPolygon2dSet(
        [ConvexPolygon2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))]
    ).transform(0, 0, 0)
    o = r.transform(0, 0, 0)
    checker = ConvexPolygon2dCollisionChecker(space, o, r)
    q = np.array([0.0, 0.0, 0.0])
    assert not checker.collisionFree(q)

    q = np.array([2.0, 0.0, 0.0])
    assert checker.collisionFree(q)

    q = np.array([0.0, 2.0, 0.0])
    assert checker.collisionFree(q)

    q = np.array([0.0, 0.0, np.pi])
    assert not checker.collisionFree(q)
