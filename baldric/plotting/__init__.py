from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from baldric.planners.planner import DiscreteGoal
from baldric.problem import Problem
from baldric.spaces import PiecewisePath
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.planners.rrt import Tree
from baldric.planners.prm import PRMPlan
from baldric.planners.rrt import RRTPlan
from loguru import logger


def plot_aabb(ax: plt.Axes, box: AABB, color="red"):
    p = patches.Rectangle(
        box._qcen - box._qlim,
        width=box._qlim[0] * 2,
        height=box._qlim[1] * 2,
        color=color,
    )
    ax.add_patch(p)


def plot_polyset(ax: plt.Axes, polyset: ConvexPolygon2dSet, color="red", alpha=1.0) -> List[patches.Polygon]:
    handles = []
    for poly in polyset.polys:
        p = patches.Polygon(poly.pts, color=color, alpha=alpha, zorder=4)
        ax.add_patch(p)
        handles.append(p)
    return handles


def plot_collision_checker(ax: plt.Axes, checker: CollisionChecker, **kwargs):
    match checker:
        case AABBCollisionChecker():
            for box in checker.boxes:
                plot_aabb(ax, box)
        case ConvexPolygon2dCollisionChecker():
            plot_polyset(ax, checker.obs)


def plot_tree(ax: plt.Axes, tree: Tree):
    for n_i, n_child_i in tree.edges:
        if n_child_i == -1:
            continue
        q_i = tree.configuration[n_i, :]
        q_c = tree.configuration[n_child_i, :]
        qs = np.vstack([q_i, q_c])
        plt.plot(qs[:, 0], qs[:, 1], "g-")
        plt.plot(qs[:, 0], qs[:, 1], "r.")


def plot_piecewise_path(ax: plt.Axes, path: PiecewisePath | None):
    if path is not None:
        qs = path.configurations
        ax.plot(qs[:, 0], qs[:, 1], "r-")
        qs = path.interpolate_with_step(1.0)
        ax.plot(qs[:, 0], qs[:, 1], "k.")


def plot_prm_plan(ax: plt.Axes, plan: PRMPlan):
    ax.plot(plan.qs[:, 0], plan.qs[:, 1], "b.")
    es = np.asarray(plan.es)

    for i in range(es.shape[0]):
        u, v = es[i, :]
        q0 = plan.qs[u]
        q1 = plan.qs[v]
        eqs = np.vstack([q0, q1])
        ax.plot(eqs[:, 0], eqs[:, 1], "g-", alpha=0.2)
    plot_piecewise_path(ax, plan.path)


def plot_rrt_plan(ax: plt.Axes, plan: RRTPlan, show_graph: bool):
    if show_graph:
        qs = plan.t.activeConfigurations
        print("# configurations", qs.shape)
        ax.plot(qs[:, 0], qs[:, 1], "b.")
        for i, j in plan.t.edges:
            q0 = qs[i, :]
            q1 = qs[j, :]
            eqs = np.vstack([q0, q1])
            ax.plot(eqs[:, 0], eqs[:, 1], "g-", alpha=0.4)
    plot_piecewise_path(ax, plan.path)


def plot_path_configurations(ax: plt.Axes, path: PiecewisePath, bot: ConvexPolygon2dSet):
    qs = path.configurations
    for i in range(path.nconfigurations):
        q = qs[i]
        plot_polyset(ax, bot.transform(q), color="grey", alpha=0.4)


def plot_without_solve(problem: Problem, dst: str):
    fig = plt.figure()
    plt.tight_layout()
    ax = plt.gca()
    plot_collision_checker(ax, problem.collision_checker)
    match problem.collision_checker:
        case ConvexPolygon2dCollisionChecker():
            bot = problem.collision_checker.robot
            plot_polyset(ax, bot.transform(problem.init), "green")
    match problem.goal, problem.collision_checker:
        case DiscreteGoal(), ConvexPolygon2dCollisionChecker():
            bot = problem.collision_checker.robot
            plot_polyset(ax, bot.transform(problem.goal.location), "purple")

    lo = problem.space._low
    hi = problem.space._high
    ax.set_xlim([lo[0], hi[1]])
    ax.set_ylim([lo[1], hi[1]])
    ax.set_aspect("equal")
    plt.tight_layout(pad=0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dst, bbox_inches="tight", pad_inches=0)


def plot_problem(problem: Problem, dst: str, show_graph: bool):
    fig = plt.figure()
    ax = plt.gca()
    plot_collision_checker(ax, problem.collision_checker)

    plan = problem.planner.plan(problem.init, problem.goal)
    match plan:
        case PRMPlan():
            plot_prm_plan(ax, plan)
        case RRTPlan():
            logger.info("rrt plan type {}", show_graph)
            plot_rrt_plan(ax, plan, show_graph)
        case _:
            logger.info("unknown plan type")
    match problem.goal, problem.collision_checker:
        case DiscreteGoal(), ConvexPolygon2dCollisionChecker():
            bot = problem.collision_checker.robot
            plot_polyset(ax, bot.transform(problem.goal.location), "purple")

    match problem.collision_checker:
        case ConvexPolygon2dCollisionChecker():
            bot = problem.collision_checker.robot
            plot_polyset(ax, bot.transform(problem.init), "green")
            path = plan.path
            if plan.path is not None:
                plot_path_configurations(ax, path, bot)

    lo = problem.space._low
    hi = problem.space._high
    ax.set_xlim([lo[0], hi[1]])
    ax.set_ylim([lo[1], hi[1]])
    ax.set_aspect("equal")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(dst, bbox_inches="tight", pad_inches=0)


def plot_problem_anim(problem: Problem, dst: str):
    fig = plt.figure()
    ax = plt.gca()
    ax.grid()
    ax.set_aspect("equal")
    lo = problem.space._low
    hi = problem.space._high
    ax.set_xlim([lo[0], hi[1]])
    ax.set_ylim([lo[1], hi[1]])

    plot_collision_checker(ax, problem.collision_checker)

    plan = problem.planner.plan(problem.init, problem.goal)

    match plan:
        case PRMPlan():
            plot_prm_plan(ax, plan)
        case RRTPlan():
            plot_rrt_plan(ax, plan, False)

    checker = problem.collision_checker
    match checker:
        case ConvexPolygon2dCollisionChecker():
            bot = checker.robot
            path = plan.path
            if path is not None:
                pts = path.interpolate_with_step(0.5)
                handles = plot_polyset(ax, bot.transform(problem.init), "grey")

                def animate(i):
                    pt = pts[i, :]
                    rt = bot.transform(pt)
                    for h, p in zip(handles, rt.polys):
                        h.set_xy(p.pts)

                ani = animation.FuncAnimation(fig, animate, interval=10, frames=len(pts))
                ani.save(dst, writer="imagemagick", fps=10)
