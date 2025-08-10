from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from baldric.collision import CollisionChecker
from baldric.collision.aabb_collision import AABB, AABBCollisionChecker
from baldric.collision.convex_collision import (
    ConvexPolygon2d,
    ConvexPolygon2dSet,
    ConvexPolygon2dCollisionChecker,
)


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
