"""
Plotting for RelativisticOrbit.

The Schwarzschild horizon is always drawn at its true coordinate radius.  A
separate fixed-size central marker keeps the origin visible in weak-field plots
without geometrically enlarging the horizon.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from driver_relativistic_orbit import RelativisticOrbitResult
from physics_relativistic_orbit import HORIZON_RADIUS, ISCO_RADIUS


def plot_relativistic_orbit(
    result: RelativisticOrbitResult,
    *,
    show_isco: bool = True,
    show_periapsides: bool = False,
) -> None:
    """Plot the orbit in the x-y diagram coordinates."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        result.x,
        result.y,
        linewidth=1.2,
        label="Orbit",
    )

    # Fixed-size origin marker keeps the origin visible without changing the
    # physical size of the horizon.
    ax.plot(
        [0.0],
        [0.0],
        marker="o",
        linestyle="none",
        markersize=4,
        label="Central mass",
    )

    if result.model == "schwarzschild":
        horizon_patch = mpatches.Circle(
            (0.0, 0.0),
            radius=HORIZON_RADIUS,
            fill=False,
            linewidth=1.3,
            label=f"Horizon ({HORIZON_RADIUS:.0f} m)",
        )
        ax.add_patch(horizon_patch)

        if show_isco:
            isco_patch = mpatches.Circle(
                (0.0, 0.0),
                radius=ISCO_RADIUS,
                fill=False,
                linestyle="--",
                linewidth=1.0,
                label=f"ISCO ({ISCO_RADIUS:.0f} m)",
            )
            ax.add_patch(isco_patch)

    if show_periapsides and result.periapsis_indices:
        px = [result.x[i] for i in result.periapsis_indices]
        py = [result.y[i] for i in result.periapsis_indices]
        ax.plot(
            px,
            py,
            marker="x",
            linestyle="none",
            markersize=6,
            label="Periapsis",
        )

    ax.set_xlabel("x diagram coordinate (m)")
    ax.set_ylabel("y diagram coordinate (m)")
    title_model = "Schwarzschild" if result.model == "schwarzschild" else "Newtonian"
    ax.set_title(f"RelativisticOrbit — {title_model} model")
    ax.set_aspect("equal", "box")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncols=2,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()
