"""
Orbit — Newtonian test-particle motion around a fixed central mass.

The default initial conditions approximate Mercury at perihelion.  Edit the
parameters below to explore planets, satellites, comets, bound/unbound motion,
accuracy, and conservation laws.
"""

import argparse

import physics_orbit
from driver_orbit import run_orbit, OutputType
from physics_orbit import GM_SUN
from plot_orbit import plot_orbit


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Orbit",
        description="Simulate Newtonian test-particle motion around a fixed mass.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Orbit {physics_orbit.MODEL_VERSION} "
            f"(build {physics_orbit.BUILD_ID})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    parse_args()

    # Approximate Mercury-at-perihelion initial conditions.
    xInit = 4.6e10       # m
    yInit = 0.0          # m
    vxInit = 0.0         # m/s
    vyInit = 58_980.0    # m/s

    # Central gravitational parameter mu = GM.
    k = GM_SUN

    # Numerical controls.
    dt0 = 1.0e4          # maximum timestep (s)
    maxSteps = 20_000    # maximum accepted integration steps
    maxOrbits = 1.0      # accumulated azimuthal revolutions
    eps1 = 0.05          # acceleration-vector timestep tolerance
    eps2 = 1.0e-4        # corrector velocity-convergence tolerance

    # Five display modes are available:
    #   "orbit", "velocity", "position_time", "velocity_time", "energy"
    output: OutputType = "orbit"

    try:
        result = run_orbit(
            xInit=xInit,
            yInit=yInit,
            vxInit=vxInit,
            vyInit=vyInit,
            k=k,
            dt0=dt0,
            maxSteps=maxSteps,
            eps1=eps1,
            eps2=eps2,
            maxOrbits=maxOrbits,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Orbit could not run: {exc}")
        return

    reason = {
        "max_orbits": "requested revolution count reached",
        "max_steps": "maximum accepted-step count reached",
        "central_singularity": "point-mass singularity approached",
    }.get(result.termination_reason, result.termination_reason)

    print(
        f"Orbit {result.model_version} (build {result.build_id}) summary"
    )
    print(f"  termination             : {reason}")
    print(f"  accepted steps          : {result.accepted_steps}")
    print(f"  elapsed simulated time  : {result.final_time:.6g} s")
    print(f"  azimuthal revolutions   : {result.revolutions_completed:.8f}")
    print(
        "  max fractional energy drift       : "
        f"{result.max_fractional_energy_drift:.3e}"
    )
    if result.max_fractional_angular_momentum_drift is None:
        print("  max fractional angular-momentum drift: n/a (initial h is zero)")
    else:
        print(
            "  max fractional angular-momentum drift: "
            f"{result.max_fractional_angular_momentum_drift:.3e}"
        )

    if result.closure_radius_residual is not None:
        print(
            "  closure radius residual            : "
            f"{result.closure_radius_residual:.3e}"
        )
        print(
            "  closure velocity residual          : "
            f"{result.closure_velocity_residual:.3e}"
        )

    try:
        plot_orbit(result, output=output)
    except ValueError as exc:
        print(f"Orbit could not display the selected output: {exc}")


if __name__ == "__main__":
    main()
