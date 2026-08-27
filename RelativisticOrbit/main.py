"""
RelativisticOrbit: equatorial test-particle orbits around a one-solar-mass
Schwarzschild source, with a Newtonian comparison mode.

The integration parameter is the particle's proper time tau.  The displayed
x-y curve is a convenient orbital diagram built from Schwarzschild areal radius
and azimuth; x and y are not global Cartesian coordinates of flat space.
"""

import argparse
import math

import physics_relativistic_orbit
from driver_relativistic_orbit import (
    RelativisticOrbitParams,
    integrate_relativistic_orbit,
)
from plot_relativistic_orbit import plot_relativistic_orbit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="RelativisticOrbit",
        description="Compare Schwarzschild and Newtonian test-particle orbits.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"RelativisticOrbit {physics_relativistic_orbit.MODEL_VERSION} "
            f"(build {physics_relativistic_orbit.BUILD_ID})"
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Parameters — edit these values to explore different orbits
# ---------------------------------------------------------------------------
params = RelativisticOrbitParams(
    x_init=1.5e4,          # initial areal radius on +x diagram axis (m)
    u_init=1.2e8,          # initial dy/dtau (m/s); not a local measured speed
    dt=2.0e-6,             # maximum proper-time step (s)
    max_steps=6_000,       # maximum accepted integration steps
    max_orbits=10,         # maximum accumulated azimuthal revolutions
    eps1=0.05,             # acceleration-vector timestep gate
    eps2=1.0e-4,           # corrector velocity-convergence tolerance
    model="schwarzschild", # "schwarzschild" or "newtonian"
)

show_isco = True
show_periapsides = False


# ---------------------------------------------------------------------------
# Run and report
# ---------------------------------------------------------------------------
def main(argv=None):
    """Run the configured simulation and return its result."""
    parse_args(argv)
    result = integrate_relativistic_orbit(params)

    reason_text = {
        "max_orbits": "requested revolution count reached",
        "max_steps": "maximum accepted-step count reached",
        "horizon": "Schwarzschild horizon crossed",
    }[result.termination_reason]

    print(
        f"RelativisticOrbit {result.model_version} "
        f"(build {result.build_id}) summary"
    )
    print(f"  model             : {result.model}")
    print(f"  termination       : {reason_text}")
    print(f"  accepted steps    : {result.final_step}")
    print(f"  proper time       : {result.tau[-1]:.6g} s")
    print(f"  azimuthal turns   : {result.n_orbits:.6f}")
    print(f"  periapsides found : {len(result.periapsis_indices)}")
    print(f"  max |Δh/h0|       : {result.max_fractional_h_drift:.3e}")
    print(f"  max |ΔE/E0|       : {result.max_fractional_energy_drift:.3e}")

    if result.mean_periapsis_advance is not None:
        degrees = math.degrees(result.mean_periapsis_advance)
        print(
            "  mean periapsis advance per radial period: "
            f"{result.mean_periapsis_advance:.6g} rad = {degrees:.6g} deg"
        )

    plot_relativistic_orbit(
        result,
        show_isco=show_isco,
        show_periapsides=show_periapsides,
    )
    return result


if __name__ == "__main__":
    result = main()
