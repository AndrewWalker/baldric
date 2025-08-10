import numpy as np
from baldric.spaces import VectorSpace
from baldric.collision import CollisionChecker
from baldric.sampler import EmbeddingFreespaceSampler


def test_sample_2d():
    low = np.array([0.0, 0.0])
    high = np.array([1.0, 1.0])
    space = VectorSpace(2)
    checker = CollisionChecker(space)
    s = EmbeddingFreespaceSampler(low=low, high=high, colltest=checker)
    q = s.sample()
    assert q.shape == (space.dimension,)


def test_sample_3d():
    low = np.array([0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0])
    space = VectorSpace(3)
    checker = CollisionChecker(space)
    s = EmbeddingFreespaceSampler(low=low, high=high, colltest=checker)
    q = s.sample()
    assert q.shape == (space.dimension,)
