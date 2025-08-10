from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)
from baldric.planners.rrt import Tree


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
            for o in checker.obs:
                plot_polyset(ax, o)


def plot_tree(ax, tree: Tree):
    for n_i, n_child_i in tree.edges:
        if n_child_i == -1:
            continue
        q_i = tree.configuration[n_i, :]
        q_c = tree.configuration[n_child_i, :]
        qs = np.vstack([q_i, q_c])
        plt.plot(qs[:, 0], qs[:, 1], "b-")
        plt.plot(qs[:, 0], qs[:, 1], "r.")
