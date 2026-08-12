"""
RelativisticOrbit plot module (Gravity From the Ground Up)

Renders the orbital trajectory and the Schwarzschild event horizon.

The horizon radius (~2,953 m for one solar mass) is many orders of magnitude
smaller than a typical orbital radius (~1e11 m).  To keep it visible at any
zoom level the circle is drawn with a radius equal to:

    max(HORIZON_RADIUS, equivalent_of_2_pixels_in_data_units)

so it always renders as at least a 2-pixel-wide colored dot while still
being a truthful to-scale circle whenever the orbit is close enough to the
horizon for it to matter (e.g. plunge orbits).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from driver_relativistic_orbit import RelativisticOrbitResult
from physics_relativistic_orbit import HORIZON_RADIUS


def plot_relativistic_orbit(result: RelativisticOrbitResult) -> None:
    """
    Plot the relativistic orbit with the Schwarzschild horizon.

    The legend is placed outside the axes area (below the plot) so it
    never obscures the trajectory.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    # ── Orbit trajectory ────────────────────────────────────────────────────
    ax.plot(result.x, result.y,
            color="royalblue", linewidth=1.2, label="Orbit")

    # ── Horizon circle ──────────────────────────────────────────────────────
    # Compute the orbit's data range so we can express "2 pixels" in data
    # coordinates.  fig.dpi gives dots-per-inch; figwidth gives inches.
    x_span = max(result.x) - min(result.x)
    y_span = max(result.y) - min(result.y)
    data_span = max(x_span, y_span)

    fig_width_px  = fig.get_figwidth()  * fig.dpi   # e.g. 7 in × 100 dpi = 700 px
    fig_height_px = fig.get_figheight() * fig.dpi

    # 2-pixel minimum radius in data units (uses the tighter of the two axes)
    min_radius = data_span * 2.0 / min(fig_width_px, fig_height_px)
    plot_radius = max(HORIZON_RADIUS, min_radius)

    horizon_patch = mpatches.Circle(
        (0.0, 0.0),
        radius=plot_radius,
        color="red",
        zorder=5,
        label=f"Horizon  ($r_g$ = {HORIZON_RADIUS:.0f} m)",
    )
    ax.add_patch(horizon_patch)

    # ── Axes formatting ─────────────────────────────────────────────────────
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title("RelativisticOrbit — Schwarzschild test particle")
    ax.set_aspect("equal", "box")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    # ── Legend outside the plot area (centred below the axes) ───────────────
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncols=2,
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()
