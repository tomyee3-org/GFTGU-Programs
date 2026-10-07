"""
Atmosphere plot module

Takes structured output from driver_atmosphere and produces graphs.
"""

import math

import matplotlib.pyplot as plt

from driver_atmosphere import CurveData


def plot_atmosphere(curve_data: CurveData, *, log: bool = False):
    """Draw the curve and open a plot window.

    curve_data is the CurveData returned by extract_output():
        x, y, y_unit, x_label, y_label, title
    """
    save_and_maybe_show(curve_data, save_path=None, show=True, log=log)


def save_and_maybe_show(
    curve_data: CurveData,
    save_path: str | None = None,
    show: bool = True,
    *,
    log: bool = False,
) -> None:
    """Draw the curve, optionally save/show it, using a log quantity axis if requested.

    Altitude stays linear. Nonpositive values are masked on the log axis;
    the original curve data are never changed.
    """
    if log and not any(math.isfinite(value) and value > 0 for value in curve_data.y):
        raise ValueError("A logarithmic quantity axis needs a finite positive value.")
    fig, ax = plt.subplots()
    ax.plot(curve_data.x, curve_data.y)
    if log:
        ax.set_yscale("log", nonpositive="mask")
    ax.set_xlabel(curve_data.x_label)
    ax.set_ylabel(curve_data.y_label)
    ax.set_title(curve_data.title + (" (log scale)" if log else ""))
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    else:
        plt.close(fig)
