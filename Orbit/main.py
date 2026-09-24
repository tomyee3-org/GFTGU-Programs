"""Command-line entry point for Orbit.

Orbit follows a Newtonian test particle around a fixed point mass.  The
student-facing input parameters are:

``xInit``, ``yInit``
    Initial Cartesian position components in metres.
``vxInit``, ``vyInit``
    Initial Cartesian velocity components in metres per second.
``k``
    Central gravitational parameter GM in m^3/s^2, not mass in kilograms.
``dt0``
    Positive maximum timestep in seconds; the adaptive solver may reduce it.
``maxSteps``
    Positive limit on accepted integration steps.
``eps1``
    Positive acceleration-vector change tolerance used to reject a trial step.
``eps2``
    Positive convergence tolerance for the iterated velocity correction.
``maxOrbits``
    Positive requested azimuthal travel, measured in revolutions.
``output``
    Plot selector. ``orbit`` shows x versus y; ``velocity`` shows the Hamilton
    hodograph; ``position_time`` and ``velocity_time`` show Cartesian
    components versus time; ``energy`` shows kinetic, potential, and total
    specific energy.

Examples
--------
  # Default Mercury-like orbit
  python main.py

  # Hyperbolic escape stopped by the accepted-step limit
  python main.py --vyInit 85000 --maxSteps 600 --output orbit

  # Moon-like test particle around a fixed Earth
  python main.py --xInit 3.626e8 --vyInit 1082 --k 3.986e14 \
    --dt0 1000 --output position_time
"""

from __future__ import annotations

import argparse
import math

import physics_orbit
from driver_orbit import OutputType, OrbitResult, TerminationReason, run_orbit
from physics_orbit import GM_SUN
from plot_orbit import plot_orbit


OUTPUT_CHOICES: tuple[OutputType, ...] = (
    "orbit",
    "velocity",
    "position_time",
    "velocity_time",
    "energy",
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, which defines every option and default."""
    parser = argparse.ArgumentParser(
        prog="Orbit",
        description=(
            "Integrate Newtonian test-particle motion around a fixed point "
            "mass and display one of five trajectory diagnostics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Orbit {physics_orbit.MODEL_VERSION} "
            f"(build {physics_orbit.BUILD_ID})"
        ),
    )

    initial = parser.add_argument_group("Initial state")
    initial.add_argument(
        "--xInit", type=float, default=4.6e10, metavar="METRES",
        help="initial x-position [m]; default approximates Mercury perihelion",
    )
    initial.add_argument(
        "--yInit", type=float, default=0.0, metavar="METRES",
        help="initial y-position [m]",
    )
    initial.add_argument(
        "--vxInit", type=float, default=0.0, metavar="M_PER_S",
        help="initial x-velocity [m/s]",
    )
    initial.add_argument(
        "--vyInit", type=float, default=58_980.0, metavar="M_PER_S",
        help="initial y-velocity [m/s]; default approximates Mercury at perihelion",
    )
    initial.add_argument(
        "--k", type=float, default=GM_SUN, metavar="M3_PER_S2",
        help="central gravitational parameter GM [m^3/s^2], not mass [kg]",
    )

    controls = parser.add_argument_group("Integration controls")
    controls.add_argument(
        "--dt0", type=float, default=1.0e4, metavar="SECONDS",
        help="positive maximum timestep; adaptive steps may be smaller [s]",
    )
    controls.add_argument(
        "--maxSteps", type=int, default=20_000, metavar="N",
        help="positive maximum number of accepted integration steps",
    )
    controls.add_argument(
        "--eps1", type=float, default=0.05, metavar="TOL",
        help="positive acceleration-vector change tolerance for trial steps",
    )
    controls.add_argument(
        "--eps2", type=float, default=1.0e-4, metavar="TOL",
        help="positive convergence tolerance for the iterated correction",
    )
    controls.add_argument(
        "--maxOrbits", type=float, default=1.0, metavar="REV",
        help="positive requested accumulated azimuthal revolutions",
    )

    display = parser.add_argument_group("Display")
    display.add_argument(
        "--output", choices=OUTPUT_CHOICES, default="orbit",
        help=(
            "plot selector: orbit=position path; velocity=Hamilton hodograph; "
            "position_time=position components; velocity_time=velocity "
            "components; energy=specific-energy components"
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _five_significant(value: float) -> str:
    """Format a finite real value with exactly five significant digits."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("summary values must be finite")
    if value == 0.0:
        return "0.0000"

    # Take the exponent from the value after rounding to five digits, so a
    # value such as 0.99999996 prints as 1.0000 rather than 1.00000.
    scientific = f"{value:.4e}"
    exponent = int(scientific.split("e")[1])
    if -4 <= exponent < 5:
        decimal_places = max(0, 4 - exponent)
        return f"{value:.{decimal_places}f}"
    return scientific


TERMINATION_TEXT: dict[TerminationReason, str] = {
    TerminationReason.MAX_ORBITS: "requested revolution count reached",
    TerminationReason.MAX_STEPS: "maximum accepted-step count reached",
    TerminationReason.CENTRAL_SINGULARITY: "point-mass singularity approached",
}


def _summary_lines(result: OrbitResult) -> list[str]:
    """Return the complete human-readable run summary."""
    reason = TERMINATION_TEXT.get(
        result.termination_reason, str(result.termination_reason)
    )

    lines = [
        f"  termination             : {reason}",
        f"  accepted steps          : {result.accepted_steps}",
        f"  elapsed simulated time  : {_five_significant(result.final_time)} s",
        "  azimuthal revolutions   : "
        f"{_five_significant(result.revolutions_completed)}",
        f"  angular-step rejections : {result.angular_step_rejections}",
        "  endpoint refinement trials: "
        f"{result.event_refinement_trials}",
        f"  acceleration evaluations : {result.acceleration_evaluations}",
    ]
    if result.shortest_accepted_step is None:
        lines.append("  accepted timestep range  : n/a (no step accepted)")
    else:
        lines.append(
            "  shortest accepted step   : "
            f"{_five_significant(result.shortest_accepted_step)} s"
        )
        lines.append(
            "  longest accepted step    : "
            f"{_five_significant(result.longest_accepted_step)} s"
        )

    if result.max_fractional_energy_drift is None:
        lines.append(
            "  max fractional energy drift       : "
            "n/a (initial energy is zero or nearly zero)"
        )
    else:
        lines.append(
            "  max fractional energy drift       : "
            f"{_five_significant(result.max_fractional_energy_drift)}"
        )
    lines.append(
        "  max absolute specific-energy drift: "
        f"{_five_significant(result.max_absolute_specific_energy_drift)} J/kg"
    )

    if result.max_fractional_angular_momentum_drift is None:
        lines.append(
            "  max fractional angular-momentum drift: "
            "n/a (initial angular momentum is zero or nearly zero)"
        )
    else:
        lines.append(
            "  max fractional angular-momentum drift: "
            f"{_five_significant(result.max_fractional_angular_momentum_drift)}"
        )
    lines.append(
        "  max absolute specific-angular-momentum drift: "
        f"{_five_significant(result.max_absolute_specific_angular_momentum_drift)} m^2/s"
    )

    if result.closure_radius_residual is None:
        lines.append("  closure radius residual            : n/a")
        lines.append("  closure velocity residual          : n/a")
    else:
        lines.append(
            "  closure radius residual            : "
            f"{_five_significant(result.closure_radius_residual)}"
        )
        lines.append(
            "  closure velocity residual          : "
            f"{_five_significant(result.closure_velocity_residual)}"
        )

    elements = result.orbital_elements
    lines.extend((
        "  Keplerian elements at initial state:",
        f"    conic classification       : {elements.classification}",
        f"    specific energy            : {_five_significant(elements.specific_energy)} J/kg",
        f"    eccentricity              : {_five_significant(elements.eccentricity)}",
    ))
    if elements.semimajor_axis is None:
        lines.append("    semimajor axis             : n/a (parabolic)")
    else:
        lines.append(
            "    semimajor axis             : "
            f"{_five_significant(elements.semimajor_axis)} m"
        )
    lines.extend((
        "    semilatus rectum           : "
        f"{_five_significant(elements.semilatus_rectum)} m",
        "    periapsis radius           : "
        f"{_five_significant(elements.periapsis_radius)} m",
    ))
    if elements.apoapsis_radius is None:
        lines.append("    apoapsis radius            : n/a (unbound)")
    else:
        lines.append(
            "    apoapsis radius            : "
            f"{_five_significant(elements.apoapsis_radius)} m"
        )
    if elements.orbital_period is None:
        lines.append("    Keplerian period           : n/a (unbound)")
    else:
        lines.append(
            "    Keplerian period           : "
            f"{_five_significant(elements.orbital_period)} s"
        )
    if elements.periapsis_longitude_degrees is None:
        lines.append("    periapsis longitude        : n/a (circular)")
        lines.append("    initial true anomaly       : n/a (circular)")
    else:
        lines.append(
            "    periapsis longitude        : "
            f"{_five_significant(elements.periapsis_longitude_degrees)} deg"
        )
        lines.append(
            "    initial true anomaly (CCW) : "
            f"{_five_significant(elements.initial_true_anomaly_degrees)} deg"
        )

    if (
        elements.classification == "hyperbolic"
        and result.termination_reason
        in (TerminationReason.MAX_STEPS, TerminationReason.MAX_ORBITS)
    ):
        final_radius = math.hypot(result.xs[-1], result.ys[-1])
        final_speed = math.hypot(result.vxs[-1], result.vys[-1])
        lines.extend((
            f"  final radius              : {_five_significant(final_radius)} m",
            f"  final speed               : {_five_significant(final_speed)} m/s",
        ))

    return lines


def main() -> None:
    args = parse_args()
    try:
        result = run_orbit(
            xInit=args.xInit,
            yInit=args.yInit,
            vxInit=args.vxInit,
            vyInit=args.vyInit,
            k=args.k,
            dt0=args.dt0,
            maxSteps=args.maxSteps,
            eps1=args.eps1,
            eps2=args.eps2,
            maxOrbits=args.maxOrbits,
        )
        lines = _summary_lines(result)
    except (OverflowError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Orbit: {exc}") from exc

    print(f"Orbit {result.model_version} (build {result.build_id}) summary")
    for line in lines:
        print(line)
    try:
        plot_orbit(result, output=args.output)
    except ValueError as exc:
        raise SystemExit(f"Orbit: {exc}") from exc


if __name__ == "__main__":
    main()
