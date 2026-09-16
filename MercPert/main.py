"""MercPert: planar circular restricted three-body experiment.

All model inputs are available at the command line (python main.py --help):
  --m_sun_solar, --m_planet_solar: masses in solar-mass units.
  --binary_separation: distance between massive bodies [m];
      --binary_separation_au accepts the same distance in AU.
  --x_init, --y_init: test-particle position RELATIVE TO SUN [m];
      --x_init_au and --y_init_au accept the same coordinates in AU.
  --vx_init, --vy_init: Sun-relative initial velocities [m/s].
  --dt: maximum working timestep [s]; --max_steps: accepted-step limit.
  --eps1: acceleration-change acceptance threshold; --eps2:
      corrector convergence threshold (0 < eps2 < eps1 < 1).
  --sun_collision_radius, --companion_collision_radius: stopping radii [m];
      zero disables that body's collision boundary.
  --position_unit: AU (astronomical units) or m (metres) on the plot.
  --annotation_corner: upper_left, upper_right, lower_left, lower_right.
  --show_jacobi_diagnostic: also display the Jacobi drift plot;
      --no-show_jacobi_diagnostic displays only the orbit plot (default).

The Sun and companion follow prescribed circular paths; the test particle has
no gravitational effect on them. Diagnostics are printed before the plots.
"""

import argparse
import math
import sys

import physics_mercpert as phys
from physics_mercpert import BinarySystemParams, MercuryInitialConditions
from driver_mercpert import MercPertRunParams, run_mercpert
from plot_mercpert import plot_jacobi_drift, plot_orbits


DEFAULTS = {
    "m_sun_solar": 1.0, "m_planet_solar": 0.1,
    "binary_separation": 0.7 * phys.AU,
    "x_init": 0.3 * phys.AU, "y_init": 0.0,
    "vx_init": 0.0, "vy_init": 59220.0,
    "dt": 2000.0, "max_steps": 10000,
    "eps1": 0.05, "eps2": 0.0001,
    "sun_collision_radius": phys.R_SUN,
    "companion_collision_radius": 0.0,
    "position_unit": "AU", "annotation_corner": "upper_left",
    "show_jacobi_diagnostic": False,
}


def finite_number(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a finite number") from error
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("enter a finite number")
    return number


def positive_number(value):
    number = finite_number(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("enter a number greater than zero")
    return number


def nonnegative_number(value):
    number = finite_number(value)
    if number < 0:
        raise argparse.ArgumentTypeError("enter a non-negative number")
    return number


def fraction(value):
    number = finite_number(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("enter a number strictly between 0 and 1")
    return number


def step_count(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("enter a positive integer")
    return number


def au_positive(value):
    number = positive_number(value) * phys.AU
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("distance in metres exceeds the numerical range")
    return number


def au_finite(value):
    number = finite_number(value) * phys.AU
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("coordinate in metres exceeds the numerical range")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="MercPert", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Explore a test particle in a circular Sun–companion binary and print orbital and numerical diagnostics.",
    )
    p.add_argument("--version", action="version",
                   version=f"MercPert {phys.MODEL_VERSION} (build {phys.BUILD_ID})")
    p.add_argument("--m_sun_solar", type=positive_number, default=DEFAULTS["m_sun_solar"],
                   help="primary mass [solar masses]")
    p.add_argument("--m_planet_solar", type=positive_number, default=DEFAULTS["m_planet_solar"],
                   help="companion mass [solar masses]; default 0.1 is a low-mass star")
    p.set_defaults(binary_separation=DEFAULTS["binary_separation"],
                   x_init=DEFAULTS["x_init"], y_init=DEFAULTS["y_init"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--binary_separation", type=positive_number, default=argparse.SUPPRESS,
                   help="Sun–companion separation [m]; default 0.7 AU")
    g.add_argument("--binary_separation_au", dest="binary_separation", type=au_positive,
                   default=argparse.SUPPRESS, help="Sun–companion separation [AU]")
    for axis in ("x", "y"):
        name = f"{axis}_init"
        g = p.add_mutually_exclusive_group()
        g.add_argument(f"--{name}", type=finite_number, default=argparse.SUPPRESS,
                       help=f"initial Sun-relative {axis} coordinate [m]; default {'0.3 AU' if axis == 'x' else '0'}")
        g.add_argument(f"--{name}_au", dest=name, type=au_finite,
                       default=argparse.SUPPRESS,
                       help=f"initial Sun-relative {axis} coordinate [AU]")
    p.add_argument("--vx_init", type=finite_number, default=DEFAULTS["vx_init"],
                   help="initial x velocity relative to the Sun [m/s]")
    p.add_argument("--vy_init", type=finite_number, default=DEFAULTS["vy_init"],
                   help="initial y velocity relative to the Sun [m/s]")
    p.add_argument("--dt", type=positive_number, default=DEFAULTS["dt"],
                   help="maximum adaptive working timestep [s]")
    p.add_argument("--max_steps", type=step_count, default=DEFAULTS["max_steps"],
                   help="maximum accepted steps; collision may stop sooner")
    p.add_argument("--eps1", type=fraction, default=DEFAULTS["eps1"],
                   help="predictor relative acceleration-change threshold (0, 1)")
    p.add_argument("--eps2", type=fraction, default=DEFAULTS["eps2"],
                   help="corrector relative velocity-change threshold (0, eps1)")
    p.add_argument("--sun_collision_radius", type=nonnegative_number,
                   default=DEFAULTS["sun_collision_radius"],
                   help="solar collision stopping radius [m]; 0 disables it")
    p.add_argument("--companion_collision_radius", type=nonnegative_number,
                   default=DEFAULTS["companion_collision_radius"],
                   help="companion collision stopping radius [m]; 0 disables it")
    p.add_argument("--position_unit", choices=("AU", "m"),
                   default=DEFAULTS["position_unit"],
                   help="orbit-axis units: AU for astronomical units; m for metres")
    p.add_argument("--annotation_corner",
                   choices=("upper_left", "upper_right", "lower_left", "lower_right"),
                   default=DEFAULTS["annotation_corner"],
                   help="corner for the Sun-relative initial-state label")
    p.add_argument("--show_jacobi_diagnostic", action=argparse.BooleanOptionalAction,
                   default=DEFAULTS["show_jacobi_diagnostic"],
                   help="show the fractional Jacobi-drift plot as well as the orbit plot")
    # argparse mistakes negative exponent notation such as -1.2e10 for a new
    # option; attach separated negative numbers to their numeric option.
    numeric = {"m_sun_solar", "m_planet_solar", "binary_separation",
               "binary_separation_au", "x_init", "x_init_au", "y_init",
               "y_init_au", "vx_init", "vy_init", "dt", "max_steps",
               "eps1", "eps2", "sun_collision_radius",
               "companion_collision_radius"}
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if (token.startswith("--") and token[2:] in numeric and index + 1 < len(raw)
                and raw[index + 1].startswith("-")
                and not raw[index + 1].startswith("--")):
            try:
                float(raw[index + 1])
            except ValueError:
                pass
            else:
                normalized.append(f"{token}={raw[index + 1]}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    args = p.parse_args(normalized)
    if args.eps2 >= args.eps1:
        p.error("--eps2 must be smaller than --eps1")
    if args.sun_collision_radius + args.companion_collision_radius >= args.binary_separation:
        p.error("collision radii must sum to less than binary separation")
    return args


def five(value):
    """Print exactly five significant digits, retaining terminal zeros."""
    return f"{value:#.5g}"


def print_summary(output, binary):
    period_days = 2 * math.pi / phys.compute_binary_angular_velocity(binary) / 86400
    distances = [math.hypot(mx - sx, my - sy) for mx, my, sx, sy in
                 zip(output.merc_x, output.merc_y, output.sun_x, output.sun_y)]
    print(f"MercPert {output.model_version} (build {output.build_id}) — "
          f"accepted steps: {output.accepted_steps}; termination: {output.termination_reason}")
    print(f"Circular period of Sun–companion pair: {five(period_days)} days")
    print(f"Test particle integration time: {five(output.times[-1] / 86400)} days "
          f"({five(output.times[-1])} s)")
    print(f"Test particle Sun-distance range: {five(min(distances) / phys.AU)} to "
          f"{five(max(distances) / phys.AU)} AU "
          f"({five(min(distances))} to {five(max(distances))} m)")
    if output.jacobi:
        c0 = output.jacobi[0]
        if c0 == 0:
            print("Maximum fractional Jacobi drift: undefined (initial Jacobi constant is zero)")
        else:
            drift = max(abs(c - c0) for c in output.jacobi) / abs(c0)
            print(f"Maximum fractional Jacobi drift: {five(drift)}")
    if output.min_working_dt is None or output.max_working_dt is None:
        print("Adaptive controller working timestep range: none (no accepted steps)")
    else:
        print("Adaptive controller working timestep range: "
              f"{five(output.min_working_dt)} to {five(output.max_working_dt)} s "
              "(accepted steps before collision truncation)")


def main(argv=None):
    args = parse_args(argv)
    binary = BinarySystemParams(args.m_sun_solar, args.m_planet_solar,
                                args.binary_separation)
    initial = MercuryInitialConditions(args.x_init, args.y_init,
                                       args.vx_init, args.vy_init)
    run = MercPertRunParams(args.dt, args.max_steps, args.eps1, args.eps2,
                            args.sun_collision_radius, args.companion_collision_radius)
    try:
        output = run_mercpert(binary, initial, run)
        print_summary(output, binary)
        plot_orbits(output, title="MercPert orbits", merc_ic=initial,
                    binary_params=binary, corner=args.annotation_corner.replace("_", " "),
                    position_unit=args.position_unit)
        if args.show_jacobi_diagnostic:
            plot_jacobi_drift(output)
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SystemExit(f"MercPert error: {error}") from error


if __name__ == "__main__":
    main()
