import numpy as np
from baldric.spaces import VectorSpace


def test_vector_space_dimension():
    space = VectorSpace.unit_box(2)
    assert space.dimension == 2


def test_vector_space_distance():
    space = VectorSpace.unit_box(2)
    q0 = np.random.random(2)
    assert np.abs(space.distance(q0, q0)) < 1e-5

    q1 = np.random.random(2)

    q0_0 = space.interpolate(q0, q1, 0.0)
    assert np.abs(space.distance(q0, q0_0)) < 1e-5

    q0_1 = space.interpolate(q0, q1, 1.0)
    assert np.abs(space.distance(q1, q0_1)) < 1e-5


def test_vector_space_interp_many():
    space = VectorSpace.unit_box(2)
    q0 = np.random.random(2)
    assert np.abs(space.distance(q0, q0)) < 1e-5

    q1 = np.random.random(2)
    qs = space.interpolate_linspace(q0, q1, 11)
    assert np.abs(space.distance(q0, qs[0])) < 1e-5
    assert np.abs(space.distance(q1, qs[-1])) < 1e-5


def test_vector_space_piecewise_path_length():
    space = VectorSpace(np.zeros(2), 3 * np.ones(2))
    path = np.array(
        [
            [1.0, 0],
            [2.0, 0],
            [2.0, 2.0],
        ]
    )
    assert np.abs(space.piecewise_path_length(path) - 3.0) < 1e-5


def test_vector_space_path_sample():
    space = VectorSpace(np.zeros(2), 3 * np.ones(2))
    path = np.array(
        [
            [1.0, 0],
            [2.0, 0],
            [2.0, 2.0],
        ]
    )
    assert np.abs(space.piecewise_path_length(path) - 3.0) < 1e-5


def test_vector_space_path_interp():
    space = VectorSpace(np.zeros(2), 3 * np.ones(2))
    path = np.array(
        [
            [1.0, 0],
            [2.0, 0],
            [2.0, 2.0],
        ]
    )
    ss = np.array([0.0, 1.0, 3.0])
    pts = space.interpolate_piecewise_path(path, ss)

    for i in range(3):
        assert np.abs(space.distance(path[i, :], pts[i, :])) < 1e-5


def test_vector_space_interp_approx_dist():
    space = VectorSpace(np.zeros(2), 20 * np.ones(2))
    q0 = np.array([1.0, 0.0])
    q1 = np.array([11.0, 0.0])
    pts = space.interpolate_approx_distance(q0, q1, 5.0)
    assert pts.shape[0] == 3
