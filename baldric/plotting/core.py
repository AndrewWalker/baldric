from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loguru import logger
from baldric.configuration import Configuration
from baldric.problem import Problem
from baldric.spaces import PiecewisePath
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABBCollisionChecker
from baldric.collision.convex_collision import ConvexPolygon2dCollisionChecker
from baldric.goals import Goal, DiscreteGoal, PredicateGoal, AABBGoal
from .geometry import plot_aabb, plot_polyset


def plot_workspace_location(ax: plt.Axes, problem: Problem, q: Configuration, alpha=0.4) -> List[patches.Patch]:
    checker = problem.collision_checker
    match checker:
        case ConvexPolygon2dCollisionChecker():
            return plot_polyset(ax, checker.robot.transform(q), alpha=alpha)
        case AABBCollisionChecker():
            return plot_aabb(ax, checker.robot.transform(q), alpha=alpha)


def plot_piecewise_path_backbone(ax: plt.Axes, path: PiecewisePath, step=1.0):
    qs = path.configurations
    ax.plot(qs[:, 0], qs[:, 1], "r-")


def plot_piecewise_path_interpolated(ax: plt.Axes, path: PiecewisePath, step=1.0):
    qs = path.interpolate_with_step(step)
    ax.plot(qs[:, 0], qs[:, 1], "k.")


def plot_piecewise_path_workspace(ax: plt.Axes, problem: Problem, path: PiecewisePath, step=1.0):
    qs = path.interpolate_with_step(step)
    for q in qs:
        plot_workspace_location(ax, problem, q)


def plot_collision_checker(ax: plt.Axes, checker: CollisionChecker, **kwargs):
    match checker:
        case AABBCollisionChecker():
            for box in checker.boxes:
                plot_aabb(ax, box)
        case ConvexPolygon2dCollisionChecker():
            plot_polyset(ax, checker.obs)


def plot_initial_workspace_location(ax: plt.Axes, problem: Problem):
    plot_workspace_location(ax, problem, problem.init)


def plot_goal(ax: plt.Axes, problem: Problem):
    g: Goal = problem.goal
    match g:
        case DiscreteGoal():
            plot_workspace_location(ax, problem, g.location)
        case PredicateGoal():
            plot_workspace_location(ax, problem, g.location)
        case AABBGoal():
            plot_aabb(ax, g.box)
        case _:
            logger.debug(f"unimplemented goal plotting {type(g)}")
