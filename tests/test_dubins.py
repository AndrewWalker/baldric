import numpy as np
from baldric.spaces import dubins


def test_simple():
    q0 = np.array([0, 0, 0])
    q1 = np.array([10, 0, 0])
    d = dubins.shortest_path(q0, q1, 1.0)
    assert abs(d.length - 10.0) < 1e-4


def test_simple_different_radius():
    q0 = np.array([0, 0, 0])
    q1 = np.array([10, 0, 0])
    d = dubins.shortest_path(q0, q1, 2.0)
    assert np.allclose(d.length, 10.0)


def test_intermediate():
    r = 1.0
    dy = 0.1

    d1 = dubins.dubins_intermediate_results((0, 0, 0), (0, -dy, 0), r)
    assert np.allclose(d1.alpha, np.pi / 2)
    assert np.allclose(d1.beta, np.pi / 2)
    assert np.allclose(d1.d, dy)

    d2 = dubins.dubins_intermediate_results((0, 0, 0), (0, dy, 0), r)
    assert np.allclose(d2.alpha, 3 * np.pi / 2)
    assert np.allclose(d2.beta, 3 * np.pi / 2)
    assert np.allclose(d2.d, dy)


def test_almost_full_loop():
    r = 1.0
    dy = 0.1
    d1 = dubins.shortest_path((0, 0, 0), (0, -dy, 0), r)
    d2 = dubins.shortest_path((0, 0, 0), (0, dy, 0), r)
    assert np.allclose(d1.length, d2.length)
    assert np.allclose(d1.length, 2 * np.pi * r + dy)


def test_half_loop():
    r = 1.0
    d = dubins.shortest_path((0, 0, 0), (0, 2 * r, -np.pi), r)
    assert np.allclose(d.length, np.pi * r)


def test_turning_radius_scaling():
    a = dubins.shortest_path((0, 0, 0), (10, 10, np.pi / 4.0), 1.0)
    b = dubins.shortest_path((0, 0, 0), (10, 10, np.pi / 4.0), 2.0)
    assert b.length > a.length
