import numpy as np
from baldric.spaces import VectorSpace
from baldric.collision import CollisionChecker
from baldric.sampler import FreespaceSampler


def test_sample_2d():
    space = VectorSpace.unit_box(2)
    checker = CollisionChecker(space)
    s = FreespaceSampler(checker)
    q = s.sample()
    assert q.shape == (space.dimension,)


def test_sample_3d():
    space = VectorSpace.unit_box(3)
    checker = CollisionChecker(space)
    s = FreespaceSampler(checker)
    q = s.sample()
    assert q.shape == (space.dimension,)
