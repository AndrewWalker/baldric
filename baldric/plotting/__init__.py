from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from baldric.problem import Problem
from baldric.collision.convex_collision import (
    ConvexPolygon2dCollisionChecker,
)
from baldric.planners import RRTPlan, PRMPlan
from loguru import logger

from .core import (
    plot_workspace_location,
    plot_piecewise_path_backbone,
    plot_piecewise_path_interpolated,
    plot_piecewise_path_workspace,
    plot_collision_checker,
    plot_initial_workspace_location,
    plot_goal,
)


def plot_prm_plan(ax: plt.Axes, plan: PRMPlan):
    """Plot the internal search structure of the plan"""
    ax.plot(plan.qs[:, 0], plan.qs[:, 1], "b.")
    es = np.asarray(plan.es)
    for i in range(es.shape[0]):
        u, v = es[i, :]
        q0 = plan.qs[u]
        q1 = plan.qs[v]
        eqs = np.vstack([q0, q1])
        ax.plot(eqs[:, 0], eqs[:, 1], "g-", alpha=0.2)


def plot_rrt_plan(ax: plt.Axes, plan: RRTPlan):
    """Plot the internal search structure of the plan"""
    qs = plan.t.activeConfigurations
    ax.plot(qs[:, 0], qs[:, 1], "b.")
    for i, j in plan.t.edges:
        q0 = qs[i, :]
        q1 = qs[j, :]
        eqs = np.vstack([q0, q1])
        ax.plot(eqs[:, 0], eqs[:, 1], "g-", alpha=0.4)


def plot_plan(ax: plt.Axes, plan: RRTPlan | PRMPlan):
    match plan:
        case RRTPlan():
            plot_rrt_plan(ax, plan)
        case PRMPlan():
            plot_prm_plan(ax, plan)


def plot_problem(problem: Problem, plan: PRMPlan | RRTPlan | None, dst: str):
    fig = plt.figure()
    plt.tight_layout()
    ax = plt.gca()

    plot_collision_checker(ax, problem.collision_checker)
    plot_initial_workspace_location(ax, problem)
    plot_goal(ax, problem)

    if plan is not None:
        plot_plan(ax, plan)
        path = plan.path
        if path is not None:
            plot_piecewise_path_backbone(ax, path)
            plot_piecewise_path_interpolated(ax, path)
            plot_piecewise_path_workspace(ax, problem, path)

    # TODO This approach is incomplete, but is suitable as a first pass
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


def animate_problem(problem: Problem, plan: PRMPlan | RRTPlan | None, dst: str):
    fig = plt.figure()
    plt.tight_layout()
    ax = plt.gca()

    plot_collision_checker(ax, problem.collision_checker)
    plot_initial_workspace_location(ax, problem)
    plot_goal(ax, problem)

    handles: List[patches.Patch] = []
    if plan is not None:
        plot_plan(ax, plan)
        path = plan.path
        if path is not None:
            handles = plot_piecewise_path_workspace(ax, problem, path)

    # TODO This approach is incomplete, but is suitable as a first pass
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


# def plot_problem_anim(problem: Problem, dst: str):
#     fig = plt.figure()
#     ax = plt.gca()
#     ax.grid()
#     ax.set_aspect("equal")
#     lo = problem.space._low
#     hi = problem.space._high
#     ax.set_xlim([lo[0], hi[1]])
#     ax.set_ylim([lo[1], hi[1]])

#     plot_collision_checker(ax, problem.collision_checker)

#     plan = problem.planner.plan(problem.init, problem.goal)

#     match plan:
#         case PRMPlan():
#             plot_prm_plan(ax, plan)
#         case RRTPlan():
#             plot_rrt_plan(ax, plan, False)

# def animate_path()
#     checker = problem.collision_checker
#     match checker:
#         case ConvexPolygon2dCollisionChecker():
#             bot = checker.robot
#             path = plan.path
#             if path is not None:
#                 pts = path.interpolate_with_step(0.5)
#                 handles = plot_polyset(ax, bot.transform(problem.init), "grey")

#                 def animate(i):
#                     pt = pts[i, :]
#                     rt = bot.transform(pt)
#                     for h, p in zip(handles, rt.polys):
#                         h.set_xy(p.pts)

#                 ani = animation.FuncAnimation(fig, animate, interval=10, frames=len(pts))
#                 ani.save(dst, writer="imagemagick", fps=10)


# from pydantic import BaseModel, Field


# class Styling(BaseModel):
#     obstacle_color: str = Field(default="grey")


# class ProblemPlottingConfig(BaseModel):
#     enable: bool
#     show_graph: bool
#     show_grid: bool
#     filename: str


# class AnimationPlottingConfig(BaseModel):
#     enable: bool
#     filename: str
#     writer: str = Field(default="imagemagick")
#     fps: int = 10


# from baldric.configuration import Configuration
# from baldric.goals import *

# def plot_robot_configuration(ax: plt.Axes, problem: Problem, q: Configuration):
#     pass


# def plot_goal(ax: plt.Axes, problem: Problem):
#     g = problem.goal
#     match g:
#         case DiscreteGoal():
#             g.
#             # plot_robot_configuration(ax, problem, g.)
#             pass
#         case PredicateGoal():
#             # plot_robot_configuration(ax, problem, g.)
#             pass
#         case AABBGoal():
#             pass


# def plot_without_solve(problem: Problem, dst: str):
#     fig = plt.figure()
#     plt.tight_layout()
#     ax = plt.gca()
#     plot_collision_checker(ax, problem.collision_checker)
#     match problem.collision_checker:
#         case ConvexPolygon2dCollisionChecker():
#             bot = problem.collision_checker.robot
#             plot_polyset(ax, bot.transform(problem.init), "green")
#     match problem.goal, problem.collision_checker:
#         case DiscreteGoal(), ConvexPolygon2dCollisionChecker():
#             bot = problem.collision_checker.robot
#             plot_polyset(ax, bot.transform(problem.goal.location), "purple")

#     lo = problem.space._low
#     hi = problem.space._high
#     ax.set_xlim([lo[0], hi[1]])
#     ax.set_ylim([lo[1], hi[1]])
#     ax.set_aspect("equal")
#     plt.tight_layout(pad=0)
#     plt.gca().set_axis_off()
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     plt.savefig(dst, bbox_inches="tight", pad_inches=0)
