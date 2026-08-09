"""
planck2_plot.py

Plotting routine for Planck2. Displays the requested spectral curve
against physical wavelength or frequency, and annotates the peak
location/value and the area under the dimensionless curve in a text
box placed in the upper right of the plot -- chosen because the curve
itself (per Investigation 10.2/10.3, Figure 10.5) rises from zero,
peaks, and falls off toward the right, leaving the upper-right corner
clear in the typical case. If a particular dataset's tail is still
prominent there, the box can be moved via the corner argument.
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt

from planck2_driver import Planck2Result

Corner = Literal["upper right", "upper left", "lower right", "lower left"]


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


def plot_planck2(result: Planck2Result, corner: Corner = "upper right",
                  y_frac_window: float = 0.003) -> None:
    """
    Plot the Planck2 result and annotate peak/area values in a text
    box in the given corner of the axes (default upper right).

    The x-min/x-max domain used for peak-finding and area integration
    (Investigation 10.3's [0.01, 100]) spans a huge dynamic range in
    physical wavelength/frequency space -- far wider than where the
    curve is actually visible. Plotting that full domain directly
    squeezes the peak into an unreadable sliver, so the displayed
    x-range is automatically windowed to where the curve is above
    `y_frac_window` times its peak value (default 0.3%), which keeps
    the full computation intact while giving a legible plot. Pass a
    smaller y_frac_window to zoom out, or 0 to show the full domain.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # x = hc/(lambda k T), Eq. 10.5, means wavelength is a *decreasing*
    # function of x -- so iterating x from x_min to x_max (as the
    # driver does) produces wavelength coordinates in decreasing
    # order, while frequency/energy-density coordinates come out
    # increasing. Sorting by physical coordinate here makes both cases
    # a clean, monotonic left-to-right curve and keeps the windowing
    # logic below correct regardless of which quantity was requested.
    coords, ys = zip(*sorted(zip(result.coord_values, result.y_values)))

    if y_frac_window > 0.0:
        threshold = y_frac_window * result.y_peak
        visible = [i for i, y in enumerate(ys) if y >= threshold]
        if visible:
            lo, hi = coords[min(visible)], coords[max(visible)]
            # Margin computed in COORDINATE space, not array-index
            # space: because wavelength/frequency is a nonlinear
            # function of the sampled x, index spacing is highly
            # non-uniform (many samples bunch up near the peak, very
            # few cover the long tail), so a small index-count margin
            # can correspond to a huge, unintended jump in physical
            # coordinate. A fractional margin of the coordinate span
            # itself avoids that.
            margin = 0.05 * (hi - lo)
            xlim = (max(coords[0], lo - margin), min(coords[-1], hi + margin))
        else:
            xlim = None
    else:
        xlim = None

    ax.plot(coords, ys, linewidth=2, color="darkred")
    ax.axvline(result.coord_peak, color="gray", linestyle=":", linewidth=1)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(result.x_label)
    ax.set_ylabel(result.y_label)
    ax.set_title(f"Planck2 — {result.quantity.replace('_', ' ').title()} "
                 f"(T = {result.T:g} K)")
    ax.grid(True, alpha=0.3)

    quantity_name = {
        "wavelength": "λ",
        "frequency": "ν",
        "energy_density": "ν",
    }[result.quantity]

    anchor = _CORNER_TO_ANCHOR[corner]
    text = (
        f"Peak at x = {result.x_peak:.4f}\n"
        f"Peak {quantity_name} = {result.coord_peak:.4e}\n"
        f"Peak value = {result.y_peak:.4e}\n"
        f"∫ f(x) dx = {result.area:.4f}"
    )
    ax.annotate(
        text,
        xy=anchor.xy, xycoords="axes fraction",
        ha=anchor.ha, va=anchor.va,
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.show()
