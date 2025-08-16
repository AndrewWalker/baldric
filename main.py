import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from baldric.sampler import FreespaceSampler
from baldric.collision.aabb_collision import AABBCollisionChecker, AABB
from baldric.collision.convex_collision import (
    ConvexPolygon2dCollisionChecker,
    ConvexPolygon2dSet,
    ConvexPolygon2d,
)
from baldric.spaces import VectorSpace, RigidBody2dSpace
from baldric.metrics import VectorNearest
from baldric.planners import Goal, DiscreteGoal
from baldric.planners.prm import PlannerPRM, PRMPlan
from baldric.plotting import plot_collision_checker, plot_polyset, plot_tree
from baldric.planners.rrt import PlannerRRT
from baldric.problem import Problem


def sample1():
    qlow = np.array([0.0, 0.0])
    qhigh = np.array([100.0, 100.0])
    p = Problem()
    p.space = VectorSpace(
        qlow,
        qhigh,
    )
    p.init = np.array([10.0, 10.0])
    p.goal = DiscreteGoal(np.array([90.0, 10.0]), 5.0, p.space)
    p.collision_checker = AABBCollisionChecker(
        space=p.space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    p.sampler = FreespaceSampler(p.collision_checker)
    p.nearest = VectorNearest(p.space)
    p.planner = PlannerPRM(p.sampler, p.nearest, p.collision_checker, r=10.0, n=1000)
    return p

    # qlow = np.array([0.0, 0.0])
    # qhigh = np.array([100.0, 100.0])
    # space = VectorSpace(2)
    # checker = AABBCollisionChecker(
    #     space=space,
    #     boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
    #     step=1.0,
    # )
    # sampler = EmbeddingFreespaceSampler(qlow, qhigh, checker)
    # nearest = VectorNearest()
    # planner = PlannerPRM(sampler, nearest, checker, r=10.0, n=1000)
    # planner.prepare()
    # q_i =
    # q_f = np.array([90.0, 10.0])
    # plan = planner.plan(q_i, q_f)

    # ax = plt.gca()
    # plot_collision_checker(ax, checker)
    # ax.set_xlim([qlow[0], qhigh[0]])
    # ax.set_ylim([qlow[1], qhigh[1]])
    # pts = space.interpolate_piecewise_path_with_step(plan.path, 5.0)
    # # plt.plot(planner.qs[:, 0], planner.qs[:, 1], "gx")
    # plt.plot(pts[:, 0], pts[:, 1], "b.")
    # plt.savefig("out.png")


def sample2():
    bot = robot()
    qlow = np.array([0.0, 0.0, -np.pi])
    qhigh = np.array([100.0, 100.0, np.pi])
    space = RigidBody2dSpace(qlow, qhigh)
    space.set_weights_from_pts(np.vstack([p.pts for p in bot.polys]))
    obs_hgt = 40.0
    obs_wid = 1.0
    obs_pts = np.array(
        [
            [50.0 - obs_wid, 0.0],
            [50.0 + obs_wid, 0.0],
            [50.0 + obs_wid, obs_hgt],
            [50.0 - obs_wid, obs_hgt],
        ]
    )
    obs = ConvexPolygon2dSet(
        polys=[
            ConvexPolygon2d(obs_pts),
            ConvexPolygon2d(obs_pts + np.array([0.0, (100 - obs_hgt)])),
        ]
    )

    checker = ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )

    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerPRM(sampler, nearest, checker, r=20.0, n=1000)
    planner.prepare()
    # q_i = np.array([10.0, 10.0, 0.0])
    # q_f = np.array([90.0, 10.0, 0.0])
    q_i = np.array([10.0, 10.0, np.pi / 2])
    q_f = np.array([10.0, 90.0, 0.0])
    plan = planner.plan(q_i, q_f)

    fig = plt.figure()
    ax = plt.gca()
    plot_collision_checker(ax, checker)
    ax.set_xlim([qlow[0], qhigh[0]])
    ax.set_ylim([qlow[1], qhigh[1]])
    pts = space.interpolate_piecewise_path_with_step(plan.path, 1.0)
    plt.plot(pts[:, 0], pts[:, 1], "g.")
    plt.plot(plan.path[:, 0], plan.path[:, 1], "kx")
    pt = pts[0, :]
    handles = plot_polyset(ax, bot.transform(pt[0], pt[1], pt[2]), color="blue")
    plt.savefig("sample2.png")

    def animate(i):
        pt = pts[i, :]
        rt = bot.transform(pt[0], pt[1], pt[2])
        for h, p in zip(handles, rt.polys):
            h.set_xy(p.pts)

    ani = animation.FuncAnimation(fig, animate, interval=20, frames=len(pts))
    ani.save("demo.gif", writer="imagemagick", fps=10)


def rrt_sample1():
    qlow = np.array([0.0, 0.0])
    qhigh = np.array([100.0, 100.0])

    space = VectorSpace(qlow, qhigh)
    checker = AABBCollisionChecker(
        space=space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerRRT(sampler, checker, nearest, n=2000, eta=3.0)
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    tree = planner.plan(q_i)

    fig = plt.figure()
    ax = plt.gca()
    plot_collision_checker(ax, checker)
    ax.set_xlim([qlow[0], qhigh[0]])
    ax.set_ylim([qlow[1], qhigh[1]])
    plot_tree(ax, tree)
    plt.savefig("rrt_sample1.png")


def rrt_sample2():
    bot = robot()
    qlow = np.array([0.0, 0.0, -np.pi])
    qhigh = np.array([100.0, 100.0, np.pi])
    space = RigidBody2dSpace(qlow, qhigh)
    space.set_weights_from_pts(np.vstack([p.pts for p in bot.polys]))
    obs_hgt = 40.0
    obs_wid = 1.0
    obs_pts = np.array(
        [
            [50.0 - obs_wid, 0.0],
            [50.0 + obs_wid, 0.0],
            [50.0 + obs_wid, obs_hgt],
            [50.0 - obs_wid, obs_hgt],
        ]
    )
    obs = ConvexPolygon2dSet(
        polys=[
            ConvexPolygon2d(obs_pts),
            ConvexPolygon2d(obs_pts + np.array([0.0, (100 - obs_hgt)])),
        ]
    )

    checker = ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )

    sampler = FreespaceSampler(checker)
    nearest = VectorNearest(space)
    planner = PlannerRRT(sampler, checker, nearest, n=1000, eta=5.0)
    q_i = np.array([10.0, 10.0, np.pi / 2])
    q_f = np.array([90.0, 10.0, 0.0])

    plan = planner.plan(q_i, DiscreteGoal(q_f, 5.0, space))

    fig = plt.figure()
    ax = plt.gca()
    plot_collision_checker(ax, checker)
    ax.set_xlim([qlow[0], qhigh[0]])
    ax.set_ylim([qlow[1], qhigh[1]])
    plot_tree(ax, plan.t)
    pth = plan.path()
    pts = space.interpolate_piecewise_path_with_step(pth, 1.0)
    plt.savefig("rrt_sample2.png")

    pt = pts[0, :]
    handles = plot_polyset(ax, bot.transform(pt[0], pt[1], pt[2]), color="blue")

    def animate(i):
        pt = pts[i, :]
        rt = bot.transform(pt[0], pt[1], pt[2])
        for h, p in zip(handles, rt.polys):
            h.set_xy(p.pts)

    ani = animation.FuncAnimation(fig, animate, interval=20, frames=len(pts))
    ani.save("rrt_demo.gif", writer="imagemagick", fps=10)


from baldric.commands.config import (
    VectorSpace2dConfig,
    ProblemConfig,
    DiscreteGoalConfig,
    AABBCheckerConfig,
    PlannerConfig,
    RRTConfig,
    PRMConfig,
)  # noqa: E402


def sample1():
    p = ProblemConfig(
        planner=PRMConfig(n=10, r=1.0),
        space=VectorSpace2dConfig(q_min=[0, 0], q_max=[10, 10]),
        checker=AABBCheckerConfig(obs=[], collsion_step=0.1),
        goal=DiscreteGoalConfig(location=[9, 1], tolerance=1.0),
        initial=[1, 1],
    )
    # p.init = np.array([10.0, 10.0])
    # p.goal = DiscreteGoal(np.array([90.0, 10.0]), 5.0, p.space)
    # p.collision_checker = AABBCollisionChecker(
    #     space=p.space,
    #     boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
    #     step=1.0,
    # )
    # p.sampler = FreespaceSampler(p.collision_checker)
    # p.nearest = VectorNearest(p.space)
    # p.planner = PlannerPRM(p.sampler, p.nearest, p.collision_checker, r=10.0, n=1000)
    return p


if __name__ == "__main__":
    import yaml

    p = sample1()
    print(yaml.safe_dump(p.model_dump()))
    # sample2()
    # rrt_sample1()
    # rrt_sample2()
