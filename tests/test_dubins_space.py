import numpy as np
from baldric.spaces.dubins_space import DubinsSpace


def test_space_basics():
    space = DubinsSpace(
        low=np.array([0, 0, -np.pi]), high=np.array([100, 100, np.pi]), rho=1.0
    )

    q0 = np.array([10, 10, np.pi / 4])
    q1 = np.array([90, 90, np.pi / 4])
    qs = np.array(
        [
            [10, 10, np.pi / 4],
            [20, 20, np.pi / 4],
        ]
    )
    space.distance(q0, q1)
    space.distance_many(qs, q1)

    space.interpolate(q0, q1, s=0.5)
