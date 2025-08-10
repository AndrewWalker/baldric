import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from baldric.sampler import EmbeddingFreespaceSampler
from baldric.collision.aabb_collision import AABBCollisionChecker, AABB
from baldric.collision.convex_collision import (
    ConvexPolygon2dCollisionChecker,
    ConvexPolygon2dSet,
    ConvexPolygon2d,
)
from baldric.spaces import VectorSpace, RigidBody2dSpace
from baldric.metrics import VectorNearest
from baldric.planners.prm import PRM, PRMPlan
from baldric.plotting import plot_collision_checker, plot_polyset, plot_tree
from baldric.planners.rrt import PlannerRRT


def robot():
    robot = [
        np.array(
            [
                [-9, 20],
                [-11, 20],
                [-11, -1],
                [-9, -1],
            ]
        ),
        np.array(
            [
                [9, 20],
                [9, -1],
                [11, -1],
                [11, 20],
            ]
        ),
        np.array(
            [
                [-10.5, 1],
                [-10.5, -1],
                [10.5, -1],
                [10.5, 1],
            ]
        ),
    ]
    return ConvexPolygon2dSet(polys=[ConvexPolygon2d(p) for p in robot])


def sample1():
    qlow = np.array([0.0, 0.0])
    qhigh = np.array([100.0, 100.0])
    space = VectorSpace(2)
    checker = AABBCollisionChecker(
        space=space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    sampler = EmbeddingFreespaceSampler(qlow, qhigh, checker)
    nearest = VectorNearest()
    planner = PRM(sampler, nearest, checker, r=10.0, n=1000)
    planner.prepare()
    q_i = np.array([10.0, 10.0])
    q_f = np.array([90.0, 10.0])
    plan = planner.plan(q_i, q_f)

    ax = plt.gca()
    plot_collision_checker(ax, checker)
    ax.set_xlim([qlow[0], qhigh[0]])
    ax.set_ylim([qlow[1], qhigh[1]])
    pts = space.interpolate_piecewise_path_with_step(plan.path, 5.0)
    # plt.plot(planner.qs[:, 0], planner.qs[:, 1], "gx")
    plt.plot(pts[:, 0], pts[:, 1], "b.")
    plt.savefig("out.png")


def sample2():
    bot = robot()
    space = RigidBody2dSpace.from_points(np.vstack([p.pts for p in bot.polys]))
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
    obs = [
        ConvexPolygon2dSet(
            polys=[
                ConvexPolygon2d(obs_pts),
                ConvexPolygon2d(obs_pts + np.array([0.0, (100 - obs_hgt)])),
            ]
        )
    ]

    checker = ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )
    qlow = np.array([0.0, 0.0, -np.pi])
    qhigh = np.array([100.0, 100.0, np.pi])
    sampler = EmbeddingFreespaceSampler(qlow, qhigh, checker)
    nearest = VectorNearest(space)
    planner = PRM(sampler, nearest, checker, r=20.0, n=1000)
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
    space = VectorSpace()
    checker = AABBCollisionChecker(
        space=space,
        boxes=[AABB(np.array([50.0, 25]), np.array([5.0, 25.0]))],
        step=1.0,
    )
    qlow = np.array([0.0, 0.0])
    qhigh = np.array([100.0, 100.0])
    sampler = EmbeddingFreespaceSampler(qlow, qhigh, checker)
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
    space = RigidBody2dSpace.from_points(np.vstack([p.pts for p in bot.polys]))
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
    obs = [
        ConvexPolygon2dSet(
            polys=[
                ConvexPolygon2d(obs_pts),
                ConvexPolygon2d(obs_pts + np.array([0.0, (100 - obs_hgt)])),
            ]
        )
    ]

    checker = ConvexPolygon2dCollisionChecker(
        space=space,
        obs=obs,
        robot=bot,
        step=1.0,
    )
    qlow = np.array([0.0, 0.0, -np.pi])
    qhigh = np.array([100.0, 100.0, np.pi])
    sampler = EmbeddingFreespaceSampler(qlow, qhigh, checker)
    nearest = VectorNearest(space)
    planner = PlannerRRT(sampler, checker, nearest, n=1000, eta=5.0)
    q_i = np.array([10.0, 10.0, np.pi / 2])
    q_f = np.array([90.0, 10.0, 0.0])

    def ingoal(q):
        return space.distance(q, q_f) < 5.0

    plan = planner.plan(q_i, ingoal)

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


if __name__ == "__main__":
    # sample1()
    # sample2()
    # rrt_sample1()
    rrt_sample2()
