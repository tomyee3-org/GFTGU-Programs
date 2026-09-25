"""RelativisticOrbit: Schwarzschild and Newtonian orbital diagrams.

Use ``python main.py --help`` for defaults and examples. Input parameters:
  --x_init: initial Schwarzschild areal radius on the +x diagram axis [m];
      outside the horizon in Schwarzschild mode.
  --u_init: initial dy/dtau [m/s], including either sign or zero; this is a
      proper-time coordinate derivative, not a locally measured speed.
  --dt: maximum proper-time integration step [s].
  --max_steps: maximum accepted steps; --max_orbits: accumulated turn limit.
  --eps1: acceleration-change gate (0 to 1); --eps2: corrector convergence
      threshold (strictly between zero and eps1).
  --model: schwarzschild (general-relativistic timelike orbit) or newtonian
      (Newtonian central force with the same initial diagram state).
  --show_isco / --no-show_isco: draw or hide the true Schwarzschild ISCO circle.
  --show_periapsides / --no-show_periapsides: mark detected radius minima.

The integration parameter is proper time tau. The x-y plot uses areal radius
and azimuth; its axes are orbital diagram coordinates, not global flat-space
Cartesian coordinates. After the run summary, the program prints the orbit
predicted from the constants of motion (physics.predict_orbit): its kind,
turning points and exact apsidal advance, an outside check on the integration.
"""

import argparse
from dataclasses import fields, replace
import math
import sys

import physics_relativistic_orbit as physics
from driver_relativistic_orbit import RelativisticOrbitParams, integrate_relativistic_orbit
from plot_relativistic_orbit import plot_relativistic_orbit


# Editable defaults retained for interactive use; CLI flags override them for
# an individual run. These are all the input fields in RelativisticOrbitParams.
params = RelativisticOrbitParams(
    x_init=1.5e4, u_init=1.2e8, dt=2.0e-6,
    max_steps=6_000, max_orbits=10, eps1=0.05, eps2=1.0e-4,
    model="schwarzschild",
)
show_isco = True
show_periapsides = False


def finite_float(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("enter a finite number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("enter a finite number")
    return value


def positive_float(text):
    value = finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("enter a number greater than zero")
    return value


def fraction(text):
    value = finite_float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("enter a number strictly between 0 and 1")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("enter a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("enter a positive integer")
    return value


def build_parser():
    """Return the command-line parser; its defaults are the current params."""
    parser = argparse.ArgumentParser(
        prog="RelativisticOrbit",
        description="Compare proper-time Schwarzschild orbits with a Newtonian central-force model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"RelativisticOrbit {physics.MODEL_VERSION} (build {physics.BUILD_ID})")
    parser.add_argument("--x_init", type=positive_float, default=params.x_init,
                        help="initial areal radius on the +x orbital diagram axis [m]; outside horizon in Schwarzschild mode")
    parser.add_argument("--u_init", type=finite_float, default=params.u_init,
                        help="initial dy/dtau [m/s], signed proper-time coordinate derivative; not local measured speed")
    parser.add_argument("--dt", type=positive_float, default=params.dt,
                        help="maximum adaptive integration step in particle proper time [s]")
    parser.add_argument("--max_steps", type=positive_int, default=params.max_steps,
                        help="maximum accepted integration steps")
    parser.add_argument("--max_orbits", type=positive_int, default=params.max_orbits,
                        help="maximum accumulated azimuthal revolutions (may overshoot slightly on the final step)")
    parser.add_argument("--eps1", type=fraction, default=params.eps1,
                        help="relative acceleration-change limit for an adaptive timestep (0 to 1)")
    parser.add_argument("--eps2", type=fraction, default=params.eps2,
                        help="corrector velocity-change threshold (0 to eps1)")
    parser.add_argument("--model", choices=("schwarzschild", "newtonian"),
                        default=params.model,
                        help="schwarzschild: general-relativistic timelike geodesic; newtonian: inverse-square central force")
    parser.add_argument("--show_isco", action=argparse.BooleanOptionalAction,
                        default=show_isco,
                        help="show the physical Schwarzschild innermost stable circular orbit circle")
    parser.add_argument("--show_periapsides", action=argparse.BooleanOptionalAction,
                        default=show_periapsides,
                        help="mark detected minimum-radius points in the orbit plot")
    return parser


def _bind_signed_numbers(raw):
    """Join a numeric option and a following negative value into one word.

    argparse normally reads a leading minus sign as an option. Binding signed
    scientific notation first lets --u_init -2e8 be read as one value.
    """
    numbers = {"x_init", "u_init", "dt", "eps1", "eps2", "max_steps", "max_orbits"}
    normalized = []
    index = 0
    while index < len(raw):
        word = raw[index]
        if (word.startswith("--") and word[2:] in numbers and index + 1 < len(raw)
                and raw[index + 1].startswith("-")
                and not raw[index + 1].startswith("--")):
            try:
                float(raw[index + 1])
            except ValueError:
                pass
            else:
                normalized.append(f"{word}={raw[index + 1]}")
                index += 2
                continue
        normalized.append(word)
        index += 1
    return normalized


def parse_args(argv=None):
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_bind_signed_numbers(raw))
    if args.eps2 >= args.eps1:
        parser.error("--eps2 must be smaller than --eps1")
    if args.model == "schwarzschild" and args.x_init <= physics.HORIZON_RADIUS:
        parser.error("--x_init must lie outside the Schwarzschild horizon")
    return args


def _angle_text(radians):
    return f"{radians:.6g} rad = {math.degrees(radians):.6g} deg"


def prediction_lines(run_params):
    """Summary lines computed from the constants of motion, not from the run."""
    prediction = physics.predict_orbit(run_params.x_init, run_params.u_init, run_params.model)
    lines = ["  predicted from the constants of motion:"]
    try:
        circular = physics.circular_proper_time_speed(run_params.x_init, run_params.model)
    except ValueError:
        lines.append("    circular dy/dtau: none (x_init is not above 3GM/c^2)")
    else:
        lines.append(f"    circular dy/dtau: {circular:.6g} m/s at x_init")
    if prediction.kind in ("bound", "circular (stable)", "circular (unstable)"):
        lines.append(f"    motion          : {prediction.kind}, periapsis "
                     f"{prediction.periapsis_radius:.6g} m, apoapsis {prediction.apoapsis_radius:.6g} m")
    elif prediction.kind == "marginal":
        lines.append("    motion          : marginal (whirls toward the unstable circular orbit at "
                     f"{prediction.periapsis_radius:.6g} m)")
    elif prediction.kind == "plunge":
        lines.append("    motion          : plunge (no inner turning point; the orbit crosses the horizon)")
    elif prediction.kind == "escape":
        lines.append("    motion          : escape (no outer turning point)")
    else:
        lines.append("    motion          : radial (zero angular momentum)")
    if prediction.apsidal_advance is not None:
        lines.append(f"    apsidal advance : {_angle_text(prediction.apsidal_advance)}")
    if prediction.weak_field_advance is not None and run_params.model.lower() == "schwarzschild":
        lines.append(f"    weak-field 6piGM/(c^2 p): {_angle_text(prediction.weak_field_advance)}")
    return lines


def main(argv=None):
    """Run the requested simulation and return its result."""
    args = parse_args(argv)
    run_params = replace(params, **{
        field.name: getattr(args, field.name)
        for field in fields(RelativisticOrbitParams)
    })
    try:
        result = integrate_relativistic_orbit(run_params)
    except (ValueError, RuntimeError, OverflowError) as exc:
        raise SystemExit(f"RelativisticOrbit: {exc}") from exc

    reason_text = {
        "max_orbits": "requested revolution count reached",
        "max_steps": "maximum accepted-step count reached",
        "horizon": "Schwarzschild horizon crossed",
    }[result.termination_reason]

    print(f"RelativisticOrbit {result.model_version} (build {result.build_id}) summary")
    print(f"  model             : {result.model}")
    if result.model == "schwarzschild":
        print(f"  horizon radius    : {physics.HORIZON_RADIUS:.5g} m")
        print(f"  ISCO radius       : {physics.ISCO_RADIUS:.5g} m")
    print(f"  termination       : {reason_text}")
    print(f"  accepted steps    : {result.final_step}")
    print(f"  proper time       : {result.tau[-1]:.6g} s")
    print(f"  azimuthal turns   : {result.n_orbits:.6f}")
    print(f"  periapsides found : {len(result.periapsis_indices)}")
    print(f"  max |Δh/h0|       : {result.max_fractional_h_drift:.3e}")
    print(f"  max |ΔE/E0|       : {result.max_fractional_energy_drift:.3e}")
    if result.periapsis_radius:
        print(f"  periapsis radius  : min {min(result.periapsis_radius):.6g} m, "
              f"max {max(result.periapsis_radius):.6g} m")
    if result.mean_periapsis_advance is not None:
        degrees = math.degrees(result.mean_periapsis_advance)
        print("  mean periapsis advance per radial period: "
              f"{result.mean_periapsis_advance:.6g} rad = {degrees:.6g} deg")
    for line in prediction_lines(run_params):
        print(line)
    plot_relativistic_orbit(result, show_isco=args.show_isco,
                            show_periapsides=args.show_periapsides)
    return result


if __name__ == "__main__":
    main()
