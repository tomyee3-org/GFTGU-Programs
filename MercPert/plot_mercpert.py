"""
MercPert plotting module

Separates plotting from physics and driver logic.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt

from driver_mercpert import MercPertOutput
from physics_mercpert import MercuryInitialConditions, AU

Corner = Literal["upper right", "upper left", "lower right", "lower left"]
PositionUnit = Literal["m", "AU"]

_CORNER_TO_XY = {
    "upper right": dict(x=0.97, y=0.97, ha="right", va="top"),
    "upper left": dict(x=0.03, y=0.97, ha="left", va="top"),
    "lower right": dict(x=0.97, y=0.03, ha="right", va="bottom"),
    "lower left": dict(x=0.03, y=0.03, ha="left", va="bottom"),
}

# Legend goes in the corner diagonally opposite wherever the parameter
# annotation is placed, so the two never overlap regardless of which
# corner is chosen for the annotation.
_OPPOSITE_CORNER = {
    "upper right": "lower left",
    "upper left": "lower right",
    "lower right": "upper left",
    "lower left": "upper right",
}


def plot_orbits(output: MercPertOutput,
                title: str = "MercPert orbits",
                merc_ic: Optional[MercuryInitialConditions] = None,
                corner: Corner = "upper left",
                position_unit: PositionUnit = "m") -> None:
    """
    Plot the orbits of Sun, Planet, and Mercury in the x-y plane.

    Every figure is rendered at the same physical size and with the same
    square axes-box shape, regardless of how the trajectory itself is
    shaped. This uses `adjustable="datalim"` rather than the default `"box"`:
    "box" would keep x/y data scaled 1:1 by shrinking the *axes box* to
    match the data's own aspect ratio, and would produce inconsistently
    sized/shaped plots. Using "datalim" instead keeps the box
    itself a fixed square and pads the data's shorter axis to fill it,
    so the box is always the same shape and only the data scale changes.

    If merc_ic is given, the initial position and velocity are annotated
    directly on the plot (velocity always in m/s; position in whichever
    unit position_unit selects).

    position_unit : "m" (default) or "AU"
        All internal physics (in physics_mercpert.py / driver_mercpert.py)
        works in SI units (meters) regardless of this setting -- this
        only controls how positions are *displayed*, on both axes and in
        the position annotation. Since initial positions are commonly
        specified as "0.65 * AU" etc. in main.py, "AU" can make the plot
        easier to relate back to the input parameters than the default
        meters, particularly at solar-system scales where the AU values
        stay close to 1 rather than needing scientific notation.
    """

    scale = AU if position_unit == "AU" else 1.0
    unit_label = "AU" if position_unit == "AU" else "m"

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([x / scale for x in output.sun_x],
            [y / scale for y in output.sun_y], color="red", label="Sun")
    ax.plot([x / scale for x in output.planet_x],
            [y / scale for y in output.planet_y], color="green", label="Planet")
    ax.plot([x / scale for x in output.merc_x],
            [y / scale for y in output.merc_y], color="blue", label="Mercury")

    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"y ({unit_label})")
    ax.set_title(title)
    ax.legend(loc=_OPPOSITE_CORNER[corner])
    ax.set_aspect("equal", adjustable="datalim")

    # Scientific notation suits meter-scale numbers; AU-scale positions
    # are already small, human-readable decimals and don't need it.
    if position_unit == "m":
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    if merc_ic is not None:
        anchor = _CORNER_TO_XY[corner]
        text = (
            f"pos: {merc_ic.x_init / scale:.4g}   {merc_ic.y_init / scale:.4g} {unit_label}\n"
            f"vel: {merc_ic.vx_init:.4g}   {merc_ic.vy_init:.4g} m/s"
        )
        ax.annotate(
            text,
            xy=(anchor["x"], anchor["y"]), xycoords="axes fraction",
            ha=anchor["ha"], va=anchor["va"],
            fontsize=9, family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    plt.tight_layout()
    plt.show()
