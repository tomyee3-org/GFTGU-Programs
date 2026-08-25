"""
Neutron: static, spherically symmetric relativistic stellar structure.

This educational model solves the Tolman-Oppenheimer-Volkoff equations for a
simple polytropic equation of state p = K*rho**gamma.
"""

import argparse

import physics_neutron
from driver_neutron import compute_neutron_star
from plot_neutron import plot_neutron, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Neutron",
        description="Compute a polytropic TOV neutron-star model.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Neutron {physics_neutron.MODEL_VERSION} "
            f"(build {physics_neutron.BUILD_ID})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()

    # Pedagogical polytropic defaults inherited from the original exercise.
    # They should not be interpreted as a modern realistic neutron-star EOS.
    gamma = 1.666667
    pC = 1.26e35          # Pa
    K = 5.3802e3          # SI units depend on gamma

    output_type = "Pressure"   # "Pressure", "Density", or "Mass"
    log_y = False              # useful for Pressure or Density

    # Numerical resolution. The nominal radial step is
    #   scale_height / steps_per_scale.
    steps_per_scale = 400
    max_steps = 200_000

    data = compute_neutron_star(
        gamma,
        pC,
        K,
        steps_per_scale=steps_per_scale,
        max_steps=max_steps,
    )

    print(
        f"Neutron {data['model_version']} (build {data['build_id']})"
    )
    print_model_summary(data)
    plot_neutron(data, output_type, log_y=log_y)
