"""
Plotting for MercPert.
"""

from typing import Literal, Optional
import math

import matplotlib.pyplot as plt

from driver_mercpert import MercPertOutput
from physics_mercpert import (
    AU,
    BinarySystemParams,
    MercuryInitialConditions,
    compute_binary_angular_velocity,
)

Corner = Literal["upper right", "upper left", "lower right", "lower left"]
PositionUnit = Literal["m", "AU"]

_CORNER_TO_XY = {
    "upper right": dict(x=0.97, y=0.97, ha="right", va="top"),
    "upper left": dict(x=0.03, y=0.97, ha="left", va="top"),
    "lower right": dict(x=0.97, y=0.03, ha="right", va="bottom"),
    "lower left": dict(x=0.03, y=0.03, ha="left", va="bottom"),
}

_OPPOSITE_CORNER = {
    "upper right": "lower left",
    "upper left": "lower right",
    "lower right": "upper left",
    "lower left": "upper right",
}


def _validate_display_options(corner: str, position_unit: str) -> None:
    if corner not in _CORNER_TO_XY:
        raise ValueError(
            "corner must be one of: upper right, upper left, "
            "lower right, lower left."
        )
    if position_unit not in ("m", "AU"):
        raise ValueError('position_unit must be either "m" or "AU".')


def plot_orbits(
    output: MercPertOutput,
    title: str = "MercPert orbits",
    merc_ic: Optional[MercuryInitialConditions] = None,
    binary_params: Optional[BinarySystemParams] = None,
    corner: Corner = "upper left",
    position_unit: PositionUnit = "m",
) -> None:
    """
    Plot barycentric inertial trajectories of the Sun, companion, and Mercury.

    Dashed red/green curves are the prescribed primary orbits. Mercury's
    integrated trajectory is solid blue. Small filled circles mark the three
    starting positions. A black x is reserved for a finite-radius collision.
    """
    _validate_display_options(corner, position_unit)

    scale = AU if position_unit == "AU" else 1.0
    unit_label = "AU" if position_unit == "AU" else "m"

    # Limit the prescribed primary reference orbits to one complete period.
    # Mercury remains plotted for the full integration duration.
    primary_end = len(output.times)

    if binary_params is not None and output.times:
        omega = compute_binary_angular_velocity(binary_params)
        period = 2.0 * math.pi / omega

        # Include the first sample at or just beyond one period so that the
        # dashed orbit visually closes once, but do not continue into a second
        # revolution where the dashes would overlap and distort.
        primary_end = len(output.times)
        for i, t in enumerate(output.times):
            if t >= period:
                primary_end = i + 1
                break

    sx = [x / scale for x in output.sun_x[:primary_end]]
    sy = [y / scale for y in output.sun_y[:primary_end]]
    px = [x / scale for x in output.planet_x[:primary_end]]
    py = [y / scale for y in output.planet_y[:primary_end]]
    mx = [x / scale for x in output.merc_x]
    my = [y / scale for y in output.merc_y]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Prescribed binary orbits are reference geometry.
    ax.plot(
        sx, sy,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Sun's orbit",
        zorder=1,
    )
    ax.plot(
        px, py,
        color="green",
        linestyle="--",
        linewidth=1.0,
        label="Companion's orbit",
        zorder=1,
    )

    # Mercury is the simulated trajectory of interest.
    ax.plot(
        mx, my,
        color="blue",
        linestyle="-",
        linewidth=1.4,
        label="Mercury",
        zorder=2,
    )

    # Start markers: intentionally omitted from the legend.
    ax.plot(sx[0], sy[0], marker="o", markersize=4.5,
            color="red", linestyle="none", zorder=3)
    ax.plot(px[0], py[0], marker="o", markersize=4.5,
            color="green", linestyle="none", zorder=3)
    ax.plot(mx[0], my[0], marker="o", markersize=4.5,
            color="blue", linestyle="none", zorder=4)

    if output.collision_body is not None:
        # Reserve the black x exclusively for a collision event.
        ax.plot(
            mx[-1], my[-1],
            marker="x",
            markersize=9,
            markeredgewidth=2,
            color="black",
            linestyle="none",
            label=f"Collision: {output.collision_body}",
            zorder=5,
        )

    ax.set_xlabel(f"barycentric x ({unit_label})")
    ax.set_ylabel(f"barycentric y ({unit_label})")
    ax.set_title(title)
    ax.legend(loc=_OPPOSITE_CORNER[corner])
    ax.set_aspect("equal", adjustable="datalim")

    if position_unit == "m":
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    if merc_ic is not None:
        anchor = _CORNER_TO_XY[corner]
        text = (
            "initial relative to Sun:\n"
            f"pos: {merc_ic.x_init / scale:.4g}   "
            f"{merc_ic.y_init / scale:.4g} {unit_label}\n"
            f"vel: {merc_ic.vx_init:.4g}   "
            f"{merc_ic.vy_init:.4g} m/s"
        )
        ax.annotate(
            text,
            xy=(anchor["x"], anchor["y"]),
            xycoords="axes fraction",
            ha=anchor["ha"],
            va=anchor["va"],
            fontsize=9,
            family="monospace",
            fontweight="bold",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )

    plt.tight_layout()
    plt.show()


def plot_jacobi_drift(output: MercPertOutput) -> None:
    """Plot fractional drift of the Jacobi constant as an accuracy diagnostic."""
    if not output.jacobi:
        raise ValueError("No Jacobi data are available.")

    c0 = output.jacobi[0]
    scale = abs(c0) if c0 != 0.0 else 1.0
    drift = [(c - c0) / scale for c in output.jacobi]
    days = [t / 86400.0 for t in output.times]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(days, drift)
    ax.set_xlabel("time (days)")
    ax.set_ylabel(r"$(C-C_0)/|C_0|$")
    ax.set_title("MercPert Jacobi-constant drift")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
