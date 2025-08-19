from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from baldric.spaces import VectorSpace, RigidBody2dSpace, PiecewisePath

unit_vec3_strategy = arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


@given(q0=unit_vec3_strategy, q1=unit_vec3_strategy)
def test_distance_symmetry(q0, q1):
    space = VectorSpace(np.zeros(3), np.ones(3))
    l1 = space.distance(q0, q1)
    l2 = space.distance(q1, q0)
    assert l1 == l2


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


@given(q0=float_array_strategy, q1=float_array_strategy)
def test_distance_symmetry_rigid2d(q0, q1):
    space = RigidBody2dSpace([0, 0, -np.pi], [100.0, 100.0, np.pi])
    l1 = space.distance(q0, q1)
    l2 = space.distance(q1, q0)
    assert l1 == l2


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


def test_vector_space_interp_approx_dist():
    space = VectorSpace(np.zeros(2), 20 * np.ones(2))
    q0 = np.array([1.0, 0.0])
    q1 = np.array([11.0, 0.0])
    # This api is required for collision tests
    pts = space.interpolate_approx_distance(q0, q1, 5.0)
    assert pts.shape[0] == 3
