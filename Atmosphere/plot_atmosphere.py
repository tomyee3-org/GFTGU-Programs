"""
Atmosphere plot module

Takes structured output from driver_atmosphere and produces graphs.
"""

import matplotlib.pyplot as plt

from driver_atmosphere import CurveData


def plot_atmosphere(curve_data: CurveData) -> None:
    """
    curve_data is the CurveData returned by extract_output():
        x, y, y_unit, x_label, y_label, title
    """
    x = curve_data.x
    y = curve_data.y
    x_label = curve_data.x_label
    y_label = curve_data.y_label
    title = curve_data.title

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)

    plt.tight_layout()
    plt.show()
