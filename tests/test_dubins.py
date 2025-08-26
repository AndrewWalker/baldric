import numpy as np
from baldric.spaces import dubins as pydubins


def test_simple():
    q0 = np.array([0, 0, 0])
    q1 = np.array([10, 0, 0])
    d = pydubins.shortest_path(q0, q1, 1.0)
    assert abs(d.length - 10.0) < 1e-4


def test_simple_different_radius():
    q0 = np.array([0, 0, 0])
    q1 = np.array([10, 0, 0])
    d = pydubins.shortest_path(q0, q1, 2.0)
    assert np.allclose(d.length, 10.0)


def test_intermediate():
    r = 1.0
    dy = 0.1

    d1 = pydubins.dubins_intermediate_results((0, 0, 0), (0, -dy, 0), r)
    assert np.allclose(d1.alpha, np.pi / 2)
    assert np.allclose(d1.beta, np.pi / 2)
    assert np.allclose(d1.d, dy)

    d2 = pydubins.dubins_intermediate_results((0, 0, 0), (0, dy, 0), r)
    assert np.allclose(d2.alpha, 3 * np.pi / 2)
    assert np.allclose(d2.beta, 3 * np.pi / 2)
    assert np.allclose(d2.d, dy)


def test_almost_full_loop():
    r = 1.0
    dy = 0.1
    d1 = pydubins.shortest_path((0, 0, 0), (0, -dy, 0), r)
    d2 = pydubins.shortest_path((0, 0, 0), (0, dy, 0), r)
    assert np.allclose(d1.length, d2.length)
    assert np.allclose(d1.length, 2 * np.pi * r + dy)


def test_half_loop():
    r = 1.0
    d = pydubins.shortest_path((0, 0, 0), (0, 2 * r, -np.pi), r)
    assert np.allclose(d.length, np.pi * r)


def test_turning_radius_scaling():
    a = pydubins.shortest_path((0, 0, 0), (10, 10, np.pi / 4.0), 1.0)
    b = pydubins.shortest_path((0, 0, 0), (10, 10, np.pi / 4.0), 2.0)
    assert b.length > a.length


from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

# Define individual element strategies for each position
element_0_1 = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
element_2 = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)

# Create array strategy with different ranges for each element
float_array_strategy = st.builds(
    np.array,
    st.tuples(
        element_0_1,  # First element: [0, 100]
        element_0_1,  # Second element: [0, 100]
        element_2,  # Third element: [-π, π]
    ),
)


def vdist(q0, q1):
    d = np.linalg.norm(q0[:2] - q1[:2])
    t = min(pydubins.mod2pi(q0[2] - q1[2]), pydubins.mod2pi(q1[2] - q0[2]))
    return d + t


@given(q0=float_array_strategy, q1=float_array_strategy)
def test_distance_symmetry_rigid2d(q0, q1):
    if np.linalg.norm(q0[:2] - q1[:2]) > 10.0:
        pth = pydubins.shortest_path(q0, q1, 1.0)
        print(pth.path_type)
        res = pydubins.dubins_path_sample(pth, 0.0)
        assert vdist(res, q0) < 1e-6

        res = pydubins.dubins_path_sample(pth, pth.length)
        print("!!!", res, q1, np.linalg.norm(res - q1))
        print(pth.param)
        assert vdist(res, q1) < 1e-6


# import dubins as cdubins


# def test_comp():
#     for i in range(50000):
#         sz = np.array([100, 100, np.pi * 2])
#         q0 = np.random.random((3,)) * sz
#         q1 = np.random.random((3,)) * sz
#         p0 = cdubins.shortest_path(q0, q1, 1.0)
#         p1 = pydubins.shortest_path(q0, q1, 1.0)
#         for w in range(6):
#             pth = cdubins.path(q0, q1, 1.0, w)
#             if pth is not None:
#                 print(w, pth.path_length())
#         dl = p0.path_length() - p1.length
#         assert abs(dl) < 1e-5
