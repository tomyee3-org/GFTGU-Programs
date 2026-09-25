"""
Atmosphere plot module

Takes structured output from driver_atmosphere and produces graphs.
"""

import matplotlib.pyplot as plt

from driver_atmosphere import CurveData


def plot_atmosphere(curve_data: CurveData):
    """Draw the curve and open a plot window.

    curve_data is the CurveData returned by extract_output():
        x, y, y_unit, x_label, y_label, title
    """
    save_and_maybe_show(curve_data, save_path=None, show=True)


def save_and_maybe_show(
    curve_data: CurveData,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Draw the curve, optionally write it to ``save_path``, optionally show it."""
    fig, ax = plt.subplots()
    ax.plot(curve_data.x, curve_data.y)
    ax.set_xlabel(curve_data.x_label)
    ax.set_ylabel(curve_data.y_label)
    ax.set_title(curve_data.title)
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    if show:
        plt.show()
    else:
        plt.close(fig)
