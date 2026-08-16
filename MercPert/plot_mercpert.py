"""
Plotting for MercPert.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt

from driver_mercpert import MercPertOutput
from physics_mercpert import AU, MercuryInitialConditions

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
    corner: Corner = "upper left",
    position_unit: PositionUnit = "m",
) -> None:
    """
    Plot barycentric inertial trajectories of the Sun, companion, and Mercury.

    If merc_ic is supplied, its annotation is explicitly labelled as the
    Sun-relative initial state, matching the meaning of the user inputs.
    """
    _validate_display_options(corner, position_unit)

    scale = AU if position_unit == "AU" else 1.0
    unit_label = "AU" if position_unit == "AU" else "m"

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([x / scale for x in output.sun_x],
            [y / scale for y in output.sun_y],
            color="red", label="Sun")
    ax.plot([x / scale for x in output.planet_x],
            [y / scale for y in output.planet_y],
            color="green", label="Companion")
    ax.plot([x / scale for x in output.merc_x],
            [y / scale for y in output.merc_y],
            color="blue", label="Mercury")

    if output.collision_body is not None:
        ax.plot(
            output.merc_x[-1] / scale,
            output.merc_y[-1] / scale,
            marker="x",
            markersize=9,
            markeredgewidth=2,
            color="black",
            linestyle="none",
            label=f"Collision: {output.collision_body}",
        )

    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"y ({unit_label})")
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
