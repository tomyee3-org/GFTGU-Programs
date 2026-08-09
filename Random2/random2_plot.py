"""
random2_plot.py

Plotting routines for Random2.
"""

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from random2_driver import Walk2DResult


# A small palette of visually distinct colors for up to a handful of
# simultaneous walk paths (matches the "roughly six" the original
# Turbo Pascal display used, so overlapping paths near the center
# stay distinguishable).
_WALK_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
]


def plot_scaled_distance(lengths: List[float], avg_dist: List[float]) -> None:
    """
    Log-log plot of average scaled distance vs number of steps --
    the same display the original Random program produces.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(lengths, avg_dist, marker='o', linestyle='-', color='blue')

    ax.set_xlabel("Number of Steps (log scale)")
    ax.set_ylabel("Scaled Distance (log scale)")
    ax.set_title("Random2: Scaled Distance vs Number of Steps")

    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()


@dataclass
class _AnchorSpec:
    """Uniformly-typed replacement for the old dict-of-mixed-types
    anchor spec: xy is a coordinate pair, ha/va are alignment strings."""
    xy: Tuple[float, float]
    ha: str
    va: str


_CORNER_TO_ANCHOR = {
    "upper right": _AnchorSpec(xy=(0.97, 0.97), ha="right", va="top"),
    "upper left": _AnchorSpec(xy=(0.03, 0.97), ha="left", va="top"),
    "lower right": _AnchorSpec(xy=(0.97, 0.03), ha="right", va="bottom"),
    "lower left": _AnchorSpec(xy=(0.03, 0.03), ha="left", va="bottom"),
}


def plot_walk2d(result: Walk2DResult, corner: str = "upper right") -> None:
    """
    Plot the star (a circle of the walk's radius) and each random walk
    path from the center outward, in a distinct color per walk. Walks
    that reach the surface continue as a straight ray in their final
    direction of travel, representing an escaping photon; walks that
    don't reach the surface within the step budget simply end where
    they stop.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    star = Circle((0, 0), result.radius, facecolor="#fff2cc",
                  edgecolor="#b5811a", linewidth=1.5, zorder=1)
    ax.add_patch(star)
    ax.plot(0, 0, marker="*", color="#b5811a", markersize=14, zorder=3)

    n_escaped = 0
    for i, walk in enumerate(result.walks):
        color = _WALK_COLORS[i % len(_WALK_COLORS)]
        xs = [p[0] for p in walk.points]
        ys = [p[1] for p in walk.points]
        ax.plot(xs, ys, color=color, linewidth=0.8, zorder=2)

        if walk.escaped:
            n_escaped += 1
            (ex, ey), (rx, ry) = walk.ray
            ax.plot([ex, rx], [ey, ry], color=color, linewidth=1.3,
                    linestyle="-", zorder=2)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mean free paths)")
    ax.set_ylabel("y (mean free paths)")
    ax.set_title("Random2: Photon Walks From a Star's Center")

    span = result.radius * 1.8
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)

    anchor = _CORNER_TO_ANCHOR[corner]
    text = (
        f"radius = {result.radius:.2f}\n"
        f"reference steps = {result.max_steps}\n"
        f"walks = {len(result.walks)}  (escaped: {n_escaped})"
    )
    ax.annotate(
        text,
        xy=anchor.xy, xycoords="axes fraction",
        ha=anchor.ha, va=anchor.va,
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.show()