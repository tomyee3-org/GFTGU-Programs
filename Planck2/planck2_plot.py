"""Plotting routine for Planck2."""

from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import math

from planck2_driver import Planck2Result

Corner = Literal["upper right", "upper left", "lower right", "lower left"]


@dataclass(frozen=True)
class _AnchorSpec:
    """Axes-fraction position and text alignment for an annotation box."""
    xy: Tuple[float, float]
    ha: str
    va: str


_CORNER_TO_ANCHOR = {
    "upper right": _AnchorSpec(xy=(0.97, 0.97), ha="right", va="top"),
    "upper left": _AnchorSpec(xy=(0.03, 0.97), ha="left", va="top"),
    "lower right": _AnchorSpec(xy=(0.97, 0.03), ha="right", va="bottom"),
    "lower left": _AnchorSpec(xy=(0.03, 0.03), ha="left", va="bottom"),
}


def _validate_result(result: Planck2Result) -> None:
    """Reject malformed or modified result data before plotting."""
    if not isinstance(result, Planck2Result):
        raise ValueError("result must be a Planck2Result instance.")

    lengths = (
        len(result.x_values),
        len(result.coord_values),
        len(result.y_values),
    )
    if lengths[0] == 0 or len(set(lengths)) != 1:
        raise ValueError("Planck2Result sample arrays must be nonempty and equal in length.")
    if not all(math.isfinite(value) and value > 0.0 for value in result.x_values):
        raise ValueError("Planck2Result x values must be finite and positive.")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in result.coord_values
    ):
        raise ValueError("Planck2Result coordinates must be finite and positive.")
    if not all(math.isfinite(value) and value >= 0.0 for value in result.y_values):
        raise ValueError("Planck2Result y values must be finite and nonnegative.")
    if not math.isfinite(result.y_peak) or result.y_peak <= 0.0:
        raise ValueError("Planck2Result y_peak must be finite and positive.")
    if not math.isfinite(result.coord_peak) or result.coord_peak <= 0.0:
        raise ValueError("Planck2Result coord_peak must be finite and positive.")
    if result.y_peak != max(result.y_values):
        raise ValueError("Planck2Result y_peak must equal a sampled maximum.")
    if result.coord_peak not in result.coord_values:
        raise ValueError("Planck2Result coord_peak must be a sampled coordinate.")


def plot_planck2(
    result: Planck2Result,
    corner: Corner = "upper right",
    y_frac_window: float = 0.003,
) -> None:
    """Plot the selected physical spectrum and annotate numerical results."""
    _validate_result(result)
    if corner not in _CORNER_TO_ANCHOR:
        allowed = ", ".join(repr(x) for x in _CORNER_TO_ANCHOR)
        raise ValueError(f"corner must be one of: {allowed}.")
    if (
        not isinstance(y_frac_window, (int, float))
        or isinstance(y_frac_window, bool)
        or not math.isfinite(y_frac_window)
        or not 0.0 <= y_frac_window <= 1.0
    ):
        raise ValueError("y_frac_window must be a finite number from 0 to 1.")

    fig, ax = plt.subplots(figsize=(9, 6))

    # A uniform x grid maps nonuniformly into physical wavelength/frequency.
    # Sort by the physical coordinate so every displayed curve runs left-to-right.
    coords, ys = zip(*sorted(zip(result.coord_values, result.y_values)))

    if y_frac_window > 0.0:
        threshold = y_frac_window * result.y_peak
        visible = [i for i, y in enumerate(ys) if y >= threshold]
        if visible:
            first = min(visible)
            last = max(visible)

            # At a threshold of 1, only the sampled peak may be visible.  Add
            # its nearest neighbors so Matplotlib receives a genuine interval
            # rather than identical left and right limits.
            if first == last:
                first = max(0, first - 1)
                last = min(len(coords) - 1, last + 1)

            lo, hi = coords[first], coords[last]
            margin = 0.05 * (hi - lo)
            xlim = (max(coords[0], lo - margin), min(coords[-1], hi + margin))
            if xlim[0] == xlim[1]:
                xlim = None
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
    ax.set_title(
        f"Planck2 — {result.quantity.replace('_', ' ').title()} "
        f"(T = {result.T:g} K)"
    )
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
        f"∫ f(x) dx = {result.dimensionless_area:.6g}\n"
        f"Physical integral = {result.physical_integral:.4e}\n"
        f"Exact physical = {result.exact_physical_integral:.4e}"
    )
    ax.annotate(
        text,
        xy=anchor.xy,
        xycoords="axes fraction",
        ha=anchor.ha,
        va=anchor.va,
        fontsize=10,
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
