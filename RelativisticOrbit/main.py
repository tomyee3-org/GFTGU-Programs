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
Cartesian coordinates.
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


def parse_args(argv=None):
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
    # argparse normally reads a leading minus sign as an option. Bind signed
    # scientific notation first so --u_init -2e8 is interpreted as one value.
    raw = list(sys.argv[1:] if argv is None else argv)
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
    args = parser.parse_args(normalized)
    if args.eps2 >= args.eps1:
        parser.error("--eps2 must be smaller than --eps1")
    if args.model == "schwarzschild" and args.x_init <= physics.HORIZON_RADIUS:
        parser.error("--x_init must lie outside the Schwarzschild horizon")
    return args


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
    print(f"  termination       : {reason_text}")
    print(f"  accepted steps    : {result.final_step}")
    print(f"  proper time       : {result.tau[-1]:.6g} s")
    print(f"  azimuthal turns   : {result.n_orbits:.6f}")
    print(f"  periapsides found : {len(result.periapsis_indices)}")
    print(f"  max |Δh/h0|       : {result.max_fractional_h_drift:.3e}")
    print(f"  max |ΔE/E0|       : {result.max_fractional_energy_drift:.3e}")
    if result.mean_periapsis_advance is not None:
        degrees = math.degrees(result.mean_periapsis_advance)
        print("  mean periapsis advance per radial period: "
              f"{result.mean_periapsis_advance:.6g} rad = {degrees:.6g} deg")
    plot_relativistic_orbit(result, show_isco=args.show_isco,
                            show_periapsides=args.show_periapsides)
    return result


if __name__ == "__main__":
    main()
