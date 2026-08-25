"""
MercPert: planar circular restricted three-body problem.

The two massive bodies follow prescribed circular orbits about their barycentre.
Mercury is a negligible-mass test particle. Its user-supplied initial position
and velocity are specified relative to the Sun.
"""

import argparse

import physics_mercpert
from physics_mercpert import (
    AU,
    R_SUN,
    BinarySystemParams,
    MercuryInitialConditions,
)
from driver_mercpert import MercPertRunParams, run_mercpert
from plot_mercpert import plot_jacobi_drift, plot_orbits


def parse_args():
    parser = argparse.ArgumentParser(
        prog="MercPert",
        description=(
            "Integrate Mercury as a test particle in a planar circular "
            "restricted three-body system using parameters defined in main.py."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(f"MercPert {physics_mercpert.MODEL_VERSION} "
                 f"(build {physics_mercpert.BUILD_ID})"),
    )
    return parser.parse_args()


def main():
    parse_args()

    # Binary system. The default companion is 0.1 solar masses:
    # about 100 Jupiter masses and therefore in the low-mass stellar regime.
    binary_params = BinarySystemParams(
        m_sun_solar=1.0,
        m_planet_solar=0.1,
        binary_separation=0.7 * AU,
    )

    # Mercury's INITIAL POSITION AND VELOCITY RELATIVE TO THE SUN.
    merc_ic = MercuryInitialConditions(
        x_init=0.3 * AU,
        y_init=0.0,
        vx_init=0.0,
        vy_init=59220.0,
    )

    run_params = MercPertRunParams(
        dt=2000.0,
        max_steps=10000,
        eps1=0.05,
        eps2=0.0001,

        # Stop if Mercury enters the nominal solar radius. This prevents a
        # trajectory from passing through the Sun without a stopping condition.
        # The companion radius is left at zero because its physical nature is
        # user-selectable.
        sun_collision_radius=R_SUN,
        companion_collision_radius=0.0,
    )

    # Plot controls
    position_unit = "AU"      # "m" or "AU"
    annotation_corner = "upper left"
    show_jacobi_diagnostic = False

    output = run_mercpert(binary_params, merc_ic, run_params)

    print(
        f"MercPert {output.model_version} (build {output.build_id}) — "
        f"accepted steps: {output.accepted_steps}; "
        f"termination: {output.termination_reason}"
    )
    if output.jacobi:
        c0 = output.jacobi[0]
        denom = abs(c0) if c0 != 0.0 else 1.0
        max_drift = max(abs(c - c0) for c in output.jacobi) / denom
        print(f"Maximum fractional Jacobi drift: {max_drift:.3e}")

    plot_orbits(
        output,
        title="MercPert orbits",
        merc_ic=merc_ic,
        binary_params=binary_params,
        corner=annotation_corner,
        position_unit=position_unit,
    )

    if show_jacobi_diagnostic:
        plot_jacobi_drift(output)


if __name__ == "__main__":
    main()
