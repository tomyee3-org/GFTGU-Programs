"""
Plotting routines for Random2.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import List, Tuple
import statistics

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from random2_driver import Walk2DResult


_WALK_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#ffe119",  # yellow
    "#f032e6",  # magenta
    "#469990",  # teal
    "#9a6324",  # brown
    "#bfef45",  # lime
    "#000075",  # navy
]


def plot_scaled_distance(
    lengths: List[float],
    avg_dist: List[float],
) -> None:
    """Plot average scaled distance against step count on log-log axes."""
    if len(lengths) != len(avg_dist) or not lengths:
        raise ValueError(
            "lengths and avg_dist must be non-empty lists of equal length."
        )
    values = [*lengths, *avg_dist]
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0.0
        for value in values
    ):
        raise ValueError("log-log plot values must be positive finite numbers.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(lengths, avg_dist, marker="o", linestyle="-")

    ax.set_xlabel("Number of Steps (log scale)")
    ax.set_ylabel("Scaled Distance (log scale)")
    ax.set_title("Random2: Scaled Distance vs Number of Steps")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()


@dataclass(frozen=True)
class _AnchorSpec:
    """Coordinates and alignment for the results annotation box."""
    xy: Tuple[float, float]
    ha: str
    va: str


_CORNER_TO_ANCHOR = {
    "upper right": _AnchorSpec(xy=(0.97, 0.97), ha="right", va="top"),
    "upper left": _AnchorSpec(xy=(0.03, 0.97), ha="left", va="top"),
    "lower right": _AnchorSpec(xy=(0.97, 0.03), ha="right", va="bottom"),
    "lower left": _AnchorSpec(xy=(0.03, 0.03), ha="left", va="bottom"),
}


def plot_walk2d(
    result: Walk2DResult,
    corner: str = "upper right",
) -> None:
    """
    Plot the circular schematic star and each fixed-step isotropic walk.

    A short straight ray beyond the boundary is a visual indication that
    scattering has stopped in the toy model; it is not a model of the
    Sun's outer layers.
    """
    if corner not in _CORNER_TO_ANCHOR:
        allowed = ", ".join(f'"{x}"' for x in _CORNER_TO_ANCHOR)
        raise ValueError(f"corner must be one of: {allowed}.")

    fig, ax = plt.subplots(figsize=(8, 8))

    star = Circle(
        (0, 0),
        result.radius,
        facecolor="#fff2cc",
        edgecolor="#b5811a",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(star)
    ax.plot(0, 0, marker="*", color="#b5811a", markersize=14, zorder=3)

    n_escaped = 0
    escaped_steps = []

    for i, walk in enumerate(result.walks):
        color = _WALK_COLORS[i % len(_WALK_COLORS)]
        xs = [p[0] for p in walk.points]
        ys = [p[1] for p in walk.points]
        ax.plot(xs, ys, color=color, linewidth=0.8, zorder=2)

        if walk.escaped:
            n_escaped += 1
            escaped_steps.append(walk.steps_taken)
            if walk.ray is not None:
                (ex, ey), (rx, ry) = walk.ray
                ax.plot(
                    [ex, rx],
                    [ey, ry],
                    color=color,
                    linewidth=1.3,
                    linestyle="-",
                    zorder=2,
                )
        elif walk.points:
            # Distinguish a safety-capped path from an escaped path whose
            # optional outgoing ray has been suppressed.
            ax.plot(
                walk.points[-1][0],
                walk.points[-1][1],
                marker="x",
                color=color,
                markersize=7,
                markeredgewidth=1.4,
                zorder=3,
            )

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (distance units)")
    ax.set_ylabel("y (distance units)")
    ax.set_title("Random2: Isotropic Random Walks in a Schematic Star")

    span = result.radius * 1.8
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)

    anchor = _CORNER_TO_ANCHOR[corner]
    text_lines = [
        f"radius = {result.radius:.2f}",
        f"mean free path = {result.mean_free_path:g}",
        f"reference steps = {result.reference_steps}",
        f"walks = {len(result.walks)}  (escaped: {n_escaped})",
    ]
    if n_escaped < len(result.walks):
        text_lines.append("x = step cap reached")
    if escaped_steps:
        text_lines.append(
            f"median escape steps = {statistics.median(escaped_steps):.0f}"
        )

    ax.annotate(
        "\n".join(text_lines),
        xy=anchor.xy,
        xycoords="axes fraction",
        ha=anchor.ha,
        va=anchor.va,
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    plt.tight_layout()
    plt.show()
