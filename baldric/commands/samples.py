from baldric.commands.config import (
    VectorSpace2dConfig,
    ProblemConfig,
    DiscreteGoalConfig,
    AABBCheckerConfig,
    RRTConfig,
    PRMConfig,
)  # noqa: E402


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


def sample_problem_configs():
    return {
        "prm_empty": prm_empty(),
        "rrt_empty": rrt_empty(),
    }
