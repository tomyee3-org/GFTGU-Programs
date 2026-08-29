"""Plotting routines for the Star program."""

import matplotlib.pyplot as plt
from driver_star import StarResult


def plot_star_structure(result: StarResult, log_y: bool = False):
    """Plot the selected stellar quantity and return its figure and axes."""
    if type(log_y) is not bool:
        raise TypeError("log_y must be a bool.")

    output_type = result.output_type

    if output_type == "pressure":
        y = result.pressure
        ylabel = "Pressure [Pa]"
    elif output_type == "density":
        y = result.density
        ylabel = "Density [kg/m³]"
    elif output_type == "temperature":
        y = result.temperature
        ylabel = "Temperature [K]"
    elif output_type == "mass":
        y = result.mass
        ylabel = "Enclosed mass [kg]"
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    # A logarithmic axis cannot display the exact zero at the interpolated
    # stellar surface.  Validate and prepare log data before creating a figure
    # so rejected calls cannot leak an unused Matplotlib figure.
    if log_y:
        if output_type == "mass":
            raise ValueError("log_y is not useful for mass because m(0) = 0.")
        pairs = [(r, val) for r, val in zip(result.radius, y) if val > 0.0]
        if not pairs:
            raise ValueError("No positive values are available for a logarithmic plot.")
        x_plot, y_plot = zip(*pairs)

    fig, ax = plt.subplots()

    if log_y:
        ax.plot(x_plot, y_plot)
        ax.set_yscale("log")
    else:
        ax.plot(result.radius, y)

    ax.set_xlabel("Radius [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Stellar structure: {output_type} vs radius")
    ax.grid(True, which="both" if log_y else "major")
    plt.tight_layout()
    plt.show()
    return fig, ax
