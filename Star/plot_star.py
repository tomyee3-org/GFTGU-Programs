"""
Plotting routines for the Star program.

This module takes the structured output from driver.integrate_star
and produces graphs of stellar structure using matplotlib.
"""

import matplotlib.pyplot as plt

from driver_star import StarResult


def plot_star_structure(result: StarResult):
    """
    Plot the chosen quantity versus radius.
    """
    radius = result.radius
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

    fig, ax = plt.subplots()
    ax.plot(radius, y)
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Stellar structure: {output_type} vs radius")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
