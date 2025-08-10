import numpy as np
from baldric.metrics import NaiveNearest, VectorNearest
from hypothesis import given, strategies as st


def test_nearest_2d():
    n = NaiveNearest()
    d = 2
    k = 1000
    qs = np.random.random((k, d))
    val = n.nearest(qs, np.random.random(d))
    assert isinstance(val, int)
    assert val >= 0
    assert val < k


def test_near_2d():
    n = NaiveNearest()
    d = 2
    k = 1000
    qs = np.random.random((k, d))
    val = n.near(qs, np.random.random(d), 100)
    assert isinstance(val, np.ndarray)
    assert val.shape == (k,)


@given(
    k=st.integers(min_value=1, max_value=1000), d=st.integers(min_value=2, max_value=4)
)
def test_comparison_between_near_metrics(k: int, d: int):
    n = NaiveNearest()
    v = VectorNearest()
    qs = np.random.random((k, d))
    q = np.random.random(d)
    res_n = n.nearest(qs, q)
    res_v = v.nearest(qs, q)
    assert res_n == res_v


@given(
    k=st.integers(min_value=1, max_value=1000),
    d=st.integers(min_value=2, max_value=4),
    r=st.floats(min_value=0.0, max_value=2.0),
)
def test_comparison_between_nearest_metrics(k: int, d: int, r: float):
    n = NaiveNearest()
    v = VectorNearest()
    qs = np.random.random((k, d))
    q = np.random.random(d)
    res_n = n.near(qs, q, r)
    res_v = v.near(qs, q, r)
    assert res_n.shape == res_v.shape
    assert np.all(res_n == res_v)
