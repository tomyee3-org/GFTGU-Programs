"""Compute the structure of a simple Newtonian polytropic star."""

import argparse

import physics_star
from driver_star import integrate_star, OutputType
from plot_star import plot_star_structure


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Star",
        description="Compute a simple Newtonian polytropic stellar model.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Star {physics_star.MODEL_VERSION} "
            f"(build {physics_star.BUILD_ID})"
        ),
    )
    return parser.parse_args()


def main():
    parse_args()

    # Default pedagogical model: chosen to give approximately solar total
    # mass and radius.  The parameters should not be interpreted as a
    # high-fidelity model of the Sun's detailed interior or composition.
    p_c = 7.158e15       # central pressure [Pa]
    T_c = 2.263e7        # central temperature [K]
    mu = 1.285           # effective mean molecular-weight parameter
    gamma = 1.36         # polytropic exponent (gamma, not polytropic index n)

    # Numerical controls.
    max_points = 2000
    steps_per_scale = 400

    # Choose: "pressure", "density", "temperature", or "mass".
    output_type: OutputType = "pressure"

    # A logarithmic y-axis is especially useful for pressure and density.
    # Do not use log_y=True with "mass", because enclosed mass is zero at r=0.
    log_y = False

    result = integrate_star(
        p_c=p_c,
        T_c=T_c,
        mu=mu,
        gamma=gamma,
        max_points=max_points,
        steps_per_scale=steps_per_scale,
        output_type=output_type,
    )

    print(f"Star {result.model_version} (build {result.build_id})")
    plot_star_structure(result, log_y=log_y)


if __name__ == "__main__":
    main()
