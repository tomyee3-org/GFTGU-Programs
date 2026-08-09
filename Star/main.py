"""
Compute the structure of a star.

This module sets example input parameters for a polytropic stellar
model (e.g. a solar-like star) and calls the driver and plot modules.
Users may overwrite these values to explore different stars.
"""

from driver_star import integrate_star, OutputType
from plot_star import plot_star_structure


def main():
    # Example parameters for a solar-like star.
    # Users are encouraged to modify these.
    p_c = 7.158e15       # central pressure [Pa] (example value)
    T_c = 2.263e7        # central temperature [K] (example value)
    mu = 1.285           # mean molecular weight (solar composition)
    gamma = 1.36         # polytropic exponent

    # Choose what to plot: "pressure", "density", "temperature", or "mass"
    output_type: OutputType = "pressure"

    result = integrate_star(
        p_c=p_c,
        T_c=T_c,
        mu=mu,
        gamma=gamma,
        max_points=2000,
        output_type=output_type,
    )

    plot_star_structure(result)


if __name__ == "__main__":
    main()
