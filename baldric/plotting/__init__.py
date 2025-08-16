from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from baldric.problem import Problem
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.planners.rrt import Tree
from baldric.planners.prm import PRMPlan


def plot_aabb(ax, box: AABB, color="red"):
    p = patches.Rectangle(
        box._qcen - box._qlim,
        width=box._qlim[0] * 2,
        height=box._qlim[1] * 2,
        color=color,
    )
    ax.add_patch(p)


def plot_polyset(ax, polyset: ConvexPolygon2dSet, color="red") -> List[patches.Polygon]:
    handles = []
    for poly in polyset.polys:
        p = patches.Polygon(poly.pts, color=color)
        ax.add_patch(p)
        handles.append(p)
    return handles


def plot_collision_checker(ax, checker: CollisionChecker, **kwargs):
    match checker:
        case AABBCollisionChecker():
            for box in checker.boxes:
                plot_aabb(ax, box)
        case ConvexPolygon2dCollisionChecker():
            plot_polyset(ax, checker.obs)


def plot_tree(ax, tree: Tree):
    for n_i, n_child_i in tree.edges:
        if n_child_i == -1:
            continue
        q_i = tree.configuration[n_i, :]
        q_c = tree.configuration[n_child_i, :]
        qs = np.vstack([q_i, q_c])
        plt.plot(qs[:, 0], qs[:, 1], "b-")
        plt.plot(qs[:, 0], qs[:, 1], "r.")


def plot_prm_plan(ax, plan: PRMPlan):
    ax.plot(plan.qs[:, 0], plan.qs[:, 1], "b.")
    es = np.asarray(plan.es)
    for i in range(es.shape[0]):
        u, v = es[i, :]
        q0 = plan.qs[u]
        q1 = plan.qs[v]
        eqs = np.vstack([q0, q1])
        ax.plot(eqs[:, 0], eqs[:, 1], "b-")
    if plan.path_indices is not None:
        ax.plot(plan.path[:, 0], plan.path[:, 1], "r-")


def plot_problem(problem: Problem, dst: str):
    maybePlan = problem.planner.plan(problem.init, problem.goal)
    if maybePlan is not None:
        fig = plt.figure()
        ax = plt.gca()
        plot_collision_checker(ax, problem.collision_checker)
        match maybePlan:
            case PRMPlan():
                plot_prm_plan(ax, maybePlan)
        lo = problem.space._low
        hi = problem.space._high
        ax.set_xlim([lo[0], hi[1]])
        ax.set_ylim([lo[1], hi[1]])
    plt.savefig(dst)
