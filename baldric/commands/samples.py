import numpy as np
from baldric.commands.config import (
    VectorSpace2dConfig,
    RigidSpace2dConfig,
    DubinsSpaceConfig,
    ProblemConfig,
    DiscreteGoalConfig,
    DubinsGoalConfig,
    AABBCheckerConfig,
    Polygon2dCheckerConfig,
    RRTConfig,
    PRMConfig,
    Polygon2dSetConfig,
    Polygon2dConfig,
    NaiveNearestConfig,
    VectorNearestConfig,
)  # noqa: E402


def box(x, y, w, h):
    assert w > 0
    assert h > 0
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def point_robot():
    L = 1.0
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=box(-L, -L, 2 * L, 2 * L))])


def bar_robot():
    L = 20.0
    W = 1.0
    polys = [box(-L / 2, -W, L, 2 * W)]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in polys])


def l_robot():
    L = 20.0
    W = 0.5
    polys = [box(-W, -W, L, 2 * W), box(-W, -W, 2 * W, L)]

    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in polys])


def u_robot():
    L = 15.0
    W = 0.5
    polys = [
        box(-L / 2, -W, L, 2 * W),
        box(-L / 2, -W, 2 * W, L),
        box(L / 2 - W, -W, 2 * W, L),
    ]

    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in polys])


def walls(scale):
    w = 0.01
    obs = [
        box(0, 0, 1.0, w),
        box(0, 1.0 - w, 1.0, w),
        box(0, 0, w, 1.0),
        box(1.0 - w, 0, w, 1.0),
    ]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p * scale) for p in obs])


def narrow_possage(scale, hgt=0.35):
    obs_hgt = hgt
    obs_wid = 0.005
    obs_pts = np.array(
        [
            [0.5 - obs_wid, 0.0],
            [0.5 + obs_wid, 0.0],
            [0.5 + obs_wid, obs_hgt],
            [0.5 - obs_wid, obs_hgt],
        ]
    )

    obs = [obs_pts, obs_pts + np.array([0.0, (1.0 - obs_hgt)])]
    w = 2 * obs_wid
    obs += [box(0, 0, 1.0, w)]
    obs += [box(0, 1.0 - w, 1.0, w)]
    obs += [box(0, 0, w, 1.0)]
    obs += [box(1.0 - w, 0, w, 1.0)]
    obs = [poly * scale for poly in obs]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p) for p in obs])


def gates(scale):
    w = 0.01
    obs = [
        box(0, 0, 1.0, w),
        box(0, 1.0 - w, 1.0, w),
        box(0, 0, w, 1.0),
        box(1.0 - w, 0, w, 1.0),
        box(0.33, 0.0, w, 0.65),
        box(0.66, 0.35, w, 0.65),
    ]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p * scale) for p in obs])


def infeasible(scale):
    w = 0.01
    obs = [
        box(0, 0, 1.0, w),
        box(0, 1.0 - w, 1.0, w),
        box(0, 0, w, 1.0),
        box(1.0 - w, 0, w, 1.0),
        box(0.5 - w / 2, 0.0, w, 1.0),
    ]
    return Polygon2dSetConfig(polys=[Polygon2dConfig(pts=p * scale) for p in obs])


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
        planner=PRMConfig(n=500, r=8),
        space=RigidSpace2dConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi]),
        checker=Polygon2dCheckerConfig(
            robot=bar_robot(),
            obstacles=narrow_possage(100.0, hgt=0.4),
            collsion_step=0.1,
        ),
        goal=DiscreteGoalConfig(location=[80, 20, 0]),
        initial=[20, 20, np.pi / 4.0],
    )
    return p


def reverse_problem(p: ProblemConfig):
    init = p.initial.copy()
    goal = p.goal.location.copy()
    p.goal.location = init
    p.initial = goal
    return p


def rrt_narrow_passage(bot, n=1500):
    p = ProblemConfig(
        planner=RRTConfig(n=1500, eta=3.0),
        space=RigidSpace2dConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi]),
        checker=Polygon2dCheckerConfig(
            robot=bot,
            obstacles=narrow_possage(100.0, hgt=0.45),
            collsion_step=0.1,
        ),
        goal=DiscreteGoalConfig(location=[80, 20, 0], tolerance=4.0),
        initial=[20, 20, np.pi / 4.0],
    )
    return p


def rrt_gates_passage(bot, n=1500):
    p = ProblemConfig(
        planner=RRTConfig(n=1500, eta=3.0),
        space=RigidSpace2dConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi]),
        checker=Polygon2dCheckerConfig(
            robot=bot,
            obstacles=gates(100.0),
            collsion_step=0.1,
        ),
        goal=DiscreteGoalConfig(location=[80, 20, 0], tolerance=4.0),
        initial=[20, 20, np.pi / 4.0],
    )
    return p


def rrt_infeasible(bot, n=500):
    p = ProblemConfig(
        planner=RRTConfig(n=1500, eta=3.0),
        space=RigidSpace2dConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi]),
        checker=Polygon2dCheckerConfig(
            robot=bot,
            obstacles=infeasible(scale=100.0),
            collsion_step=0.1,
        ),
        goal=DiscreteGoalConfig(location=[80, 50, 0], tolerance=4.0),
        initial=[20, 50, np.pi / 4.0],
    )
    return p


def rrt_narrow_passage_rev(bot):
    p = rrt_narrow_passage(bot)
    return reverse_problem(p)


def rrt_narrow_passage_infeasible(bot):
    p = rrt_narrow_passage(bot)
    return reverse_problem(p)


def dubins_rrt_empty():
    p = ProblemConfig(
        planner=RRTConfig(n=400, eta=10.0),
        metric=NaiveNearestConfig(),
        space=DubinsSpaceConfig(q_min=[0, 0, -np.pi], q_max=[100, 100, np.pi], rho=5.0),
        # checker=Polygon2dCheckerConfig(robot=point_robot(), obstacles=walls(100), collsion_step=0.1),
        checker=Polygon2dCheckerConfig(robot=bar_robot(), obstacles=narrow_possage(100), collsion_step=0.1),
        goal=DubinsGoalConfig(location=[80, 20, 0], tolerance=5.0),
        initial=[20, 20, -np.pi],
    )
    return p


def sample_problem_configs():
    return {
        "dubins_rrt_empty": dubins_rrt_empty(),
        "prm_empty": prm_empty(),
        "rrt_empty": rrt_empty(),
        "prm_narrow_passage": prm_narrow_passage(),
        "rrt_narrow_passage": rrt_gates_passage(bar_robot()),
        "rrt_narrow_passage_pt": rrt_gates_passage(point_robot()),
        "rrt_narrow_passage_rev": rrt_narrow_passage_rev(bar_robot()),
        "rrt_narrow_passage_l": rrt_narrow_passage(l_robot()),
        "rrt_narrow_passage_u": rrt_narrow_passage(u_robot()),
        "rrt_infeasible": rrt_infeasible(point_robot()),
    }
