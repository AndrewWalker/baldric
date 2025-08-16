import numpy as np
from baldric.commands.config import (
    VectorSpace2dConfig,
    RigidSpace2dConfig,
    ProblemConfig,
    DiscreteGoalConfig,
    AABBCheckerConfig,
    Polygon2dCheckerConfig,
    RRTConfig,
    PRMConfig,
    Polygon2dSetConfig,
    Polygon2dConfig,
)  # noqa: E402


def u_shaped_robot():
    polys = [
        [
            [-9.0, 20.0],
            [-11.0, 20.0],
            [-11.0, -1.0],
            [-9.0, -1.0],
        ],
        [
            [9.0, 20.0],
            [9.0, -1.0],
            [11.0, -1.0],
            [11.0, 20.0],
        ],
        [
            [-10.5, 1.0],
            [-10.5, -1.0],
            [10.5, -1.0],
            [10.5, 1.0],
        ],
    ]
    Polygon2dConfig(pts=polys[0])
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in polys])


def narrow_possage(scale):
    obs_hgt = 0.4
    obs_wid = 1.0
    obs_pts = np.array(
        [
            [0.5 - obs_wid, 0.0],
            [0.5 + obs_wid, 0.0],
            [0.5 + obs_wid, obs_hgt],
            [0.5 - obs_wid, obs_hgt],
        ]
    )
    obs = [obs_pts, obs_pts + np.array([0.0, (1.0 - obs_hgt)])]
    obs = [poly * scale for poly in obs]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in obs])


def prm_empty():
    p = ProblemConfig(
        planner=PRMConfig(n=200, r=1.0),
        space=VectorSpace2dConfig(q_min=[0, 0], q_max=[10, 10]),
        checker=AABBCheckerConfig(obs=[], collsion_step=0.1),
        goal=DiscreteGoalConfig(location=[9, 9]),
        initial=[1, 1],
    )
    return p


def rrt_empty():
    p = ProblemConfig(
        planner=RRTConfig(n=100, eta=0.5),
        space=VectorSpace2dConfig(q_min=[0, 0], q_max=[10, 10]),
        checker=AABBCheckerConfig(obs=[], collsion_step=0.1),
        goal=DiscreteGoalConfig(location=[9, 1]),
        initial=[1, 1],
    )
    return p


def prm_narrow_passage():
    p = ProblemConfig(
        planner=PRMConfig(n=200, r=1.0),
        space=RigidSpace2dConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi]),
        checker=Polygon2dCheckerConfig(
            robot=u_shaped_robot(), obstacles=narrow_possage(100.0), collsion_step=0.1
        ),
        goal=DiscreteGoalConfig(location=[9, 9, 0]),
        initial=[1, 1, 0],
    )
    return p


def sample_problem_configs():
    return {
        "prm_empty": prm_empty(),
        "rrt_empty": rrt_empty(),
        "prm_narrow_passage": prm_narrow_passage(),
    }
