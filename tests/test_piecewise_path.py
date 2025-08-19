import numpy as np
from baldric.spaces import PiecewisePath, VectorSpace


def test_create_path_2d():
    space = VectorSpace(np.zeros(2), 10 * np.ones(2))
    pts = np.array([(1, 1), (2, 1), (2, 3)], dtype=np.float64)
    path = PiecewisePath(
        space,
        pts,
    )
    assert path.length == 3.0
    assert path.nsegments == 2
    assert np.allclose(path.segment_lengths, np.array([1.0, 2.0]))
    assert np.allclose(path.cumlen, np.array([0.0, 1.0, 3.0]))
    assert np.allclose(path.interpolate(np.array([0.0])), pts[0])
    assert np.allclose(path.interpolate(np.array([1.0])), pts[1])
    assert np.allclose(path.interpolate(np.array([3.0])), pts[2])
    ss = path.length * np.linspace(0.0, 1.0, 15)
    samps = path.interpolate(ss)
    assert np.allclose(samps[0], pts[0])
    assert np.allclose(samps[-1], pts[-1])
