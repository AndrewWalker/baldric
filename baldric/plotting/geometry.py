from typing import List
import matplotlib.pyplot as plt
from matplotlib import patches
from baldric.geometry import AABB, ConvexPolygon2dSet


def plot_aabb(ax: plt.Axes, box: AABB, color="red", alpha=0.4):
    p = patches.Rectangle(
        box._qcen - box._qlim, width=box._qlim[0] * 2, height=box._qlim[1] * 2, color=color, alpha=alpha
    )
    ax.add_patch(p)
    return [p]


def plot_polyset(ax: plt.Axes, polyset: ConvexPolygon2dSet, color="red", alpha=1.0) -> List[patches.Polygon]:
    handles = []
    for poly in polyset.polys:
        p = patches.Polygon(poly.pts, color=color, alpha=alpha, zorder=4)
        ax.add_patch(p)
        handles.append(p)
    return handles
