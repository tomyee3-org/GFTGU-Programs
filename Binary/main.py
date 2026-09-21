"""Binary: Newtonian two-body trajectories, driven by command-line options.

--MA, --MB: body masses [kg].
--xInitA, --yInitA, --xInitB, --yInitB: initial coordinates [m].
--vInitA, --uInitA, --vInitB, --uInitB: initial x and y velocities [m/s].
--dt: maximum timestep [s]; --max_steps: accepted-step limit.
--eps1: predictor acceleration-change threshold; --eps2: corrector
velocity-change threshold; both strictly between zero and one.
--stop_after_one_orbit: stop at one relative turn (default);
--no-stop_after_one_orbit: run to max_steps.
--output_type: orbits (xy paths), velocity_space (velocity plane),
position_vs_time_body_a/b (coordinates versus time),
velocity_vs_time_body_a/b (velocity components versus time), or
energy_vs_time (potential, kinetic, total mechanical energies).
"""

import argparse
from bisect import bisect_left
from math import isfinite, pi
import sys

import physics_binary as phys
from driver_binary import integrate_binary
from plot_binary import plot_binary

DEFAULTS = {
    "MA": 2e30, "MB": 2e30,
    "xInitA": 4.6e10, "yInitA": 0., "xInitB": -4.6e10, "yInitB": 0.,
    "vInitA": 0., "uInitA": 13000., "vInitB": 0., "uInitB": -13000.,
    "dt": 2000., "max_steps": 10000, "eps1": .05, "eps2": 1e-4,
    "stop_after_one_orbit": True,
}
OUTPUT_TYPES = {
    "orbits": "orbits", "velocity_space": "velocity space",
    "position_vs_time_body_a": "position vs. time, body A",
    "position_vs_time_body_b": "position vs. time, body B",
    "velocity_vs_time_body_a": "velocity vs. time, body A",
    "velocity_vs_time_body_b": "velocity vs. time, body B",
    "energy_vs_time": "energy vs time",
}


def finite_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a finite number") from error
    if not isfinite(value):
        raise argparse.ArgumentTypeError("enter a finite number")
    return value


def positive_float(text):
    value = finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("enter a number greater than zero")
    return value


def tolerance(text):
    value = finite_float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("enter a number strictly between 0 and 1")
    return value


def positive_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("enter a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("enter a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="Binary", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Integrate a Newtonian binary, print orbital and energy diagnostics, and plot one selected view.",
    )
    parser.add_argument("--version", action="version",
                        version=f"Binary {phys.MODEL_VERSION} (build {phys.BUILD_ID})")
    descriptions = {
        "MA": "mass of body A [kg]", "MB": "mass of body B [kg]",
        "xInitA": "initial x coordinate of A [m]",
        "yInitA": "initial y coordinate of A [m]",
        "xInitB": "initial x coordinate of B [m]",
        "yInitB": "initial y coordinate of B [m]",
        "vInitA": "initial x velocity of A [m/s]",
        "uInitA": "initial y velocity of A [m/s]",
        "vInitB": "initial x velocity of B [m/s]",
        "uInitB": "initial y velocity of B [m/s]",
        "dt": "maximum attempted timestep [s]",
        "max_steps": "safety limit on accepted integration steps",
        "eps1": "predictor relative acceleration-change threshold (0 to 1, exclusive)",
        "eps2": "corrector relative velocity-change threshold (0 to 1, exclusive)",
    }
    for name in descriptions:
        value_type = (positive_int if name == "max_steps" else
                      tolerance if name in ("eps1", "eps2") else
                      positive_float if name in ("MA", "MB", "dt") else finite_float)
        parser.add_argument(f"--{name}", type=value_type, default=DEFAULTS[name],
                            help=descriptions[name])
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--stop_after_one_orbit", dest="stop_after_one_orbit",
                      action="store_true", default=True,
                      help="stop when the relative position completes one turn")
    # The value True belongs to --stop_after_one_orbit above.  Suppressing the
    # default here keeps --help from printing "(default: True)" beside the
    # option that turns the stop off.
    stop.add_argument("--no-stop_after_one_orbit", dest="stop_after_one_orbit",
                      action="store_false", default=argparse.SUPPRESS,
                      help="run to max_steps, including unbound trajectories")
    parser.add_argument("--output_type", choices=tuple(OUTPUT_TYPES), default="orbits",
                        help="plot: orbits = xy tracks; velocity_space = velocity plane; "
                             "position_vs_time_body_a/b = positions versus time; "
                             "velocity_vs_time_body_a/b = velocities versus time; "
                             "energy_vs_time = potential, kinetic and total energies")
    # argparse treats exponent-form negative numbers such as -4.6e10 as new
    # options when separated by a space. Preserve ordinary --name -4.6e10
    # spelling by binding recognized numeric values to their options.
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if (token.startswith("--") and token[2:] in descriptions
                and index + 1 < len(raw) and raw[index + 1].startswith("-")
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
    return parser.parse_args(normalized)


def interpolate(times, values, when):
    """Linear interpolation within the saved integration history."""
    index = bisect_left(times, when)
    if index == 0:
        return values[0]
    if index == len(times):
        return values[-1]
    if times[index] == when:
        return values[index]
    weight = (when - times[index - 1]) / (times[index] - times[index - 1])
    return values[index - 1] + weight * (values[index] - values[index - 1])


def fmt(value, unit=""):
    return "undefined" if value is None else f"{value:.5g}{unit}"


def _mass_fractions(MA, MB):
    """Return ``(MB/(MA+MB), MA/(MA+MB))`` without overflowing the mass sum."""
    total = MA + MB
    if isfinite(total):
        return MB / total, MA / total
    return 1.0 / (1.0 + MA / MB), 1.0 / (1.0 + MB / MA)


def print_summary(result, parameters):
    elements = phys.orbital_elements(*(parameters[key] for key in (
        "MA", "MB", "xInitA", "yInitA", "vInitA", "uInitA",
        "xInitB", "yInitB", "vInitB", "uInitB")))
    separations = [phys.relative_displacement(ax, ay, bx, by)[2] for ax, ay, bx, by
                   in zip(result.xA, result.yA, result.xB, result.yB)]
    days = result.times[-1] / 86400.
    print(f"Binary {result.model_version} (build {result.build_id})")
    print(f"Accepted steps: {result.accepted_steps}")
    print(f"Termination: {'one relative revolution' if result.completed_orbit else 'max_steps'}")
    print(f"Total revolutions: {fmt(abs(result.total_angle_rad) / (2 * pi))}")
    print(f"Total time: {fmt(days)} days")
    print(f"Minimum separation (sampled): {fmt(min(separations))} m")
    print(f"Maximum separation (sampled): {fmt(max(separations))} m")
    print(f"Initial Keplerian orbit: {elements.kind}; eccentricity: {fmt(elements.eccentricity)}")
    if elements.kind == "radial":
        print("Radial trajectory (zero angular momentum): apsides and period are undefined.")
    fraction_A, fraction_B = _mass_fractions(parameters["MA"], parameters["MB"])
    for label, fraction in (("A", fraction_A), ("B", fraction_B)):
        scaled = lambda number: None if number is None else fraction * number
        period_days = None if elements.period is None else elements.period / 86400.
        print(f"Body {label} about centre of mass: semi-major axis: "
              f"{fmt(scaled(elements.relative_semimajor), ' m')}; eccentricity: "
              f"{fmt(elements.eccentricity)}; period: {fmt(period_days, ' days')}")
        print(f"  Periapsis: {fmt(scaled(elements.relative_periapsis), ' m')}; "
              f"apoapsis: {fmt(scaled(elements.relative_apoapsis), ' m')}")
        print(f"  Speed at periapsis: {fmt(scaled(elements.relative_speed_periapsis), ' m/s')}; "
              f"speed at apoapsis: {fmt(scaled(elements.relative_speed_apoapsis), ' m/s')}")
    print("Energy at fractions of total run days (linear interpolation; SI joules):")
    print("Fraction     Day            Potential U         Kinetic K          Total E       (E-E0)/|E0|")
    initial = result.E[0]
    for j in range(11):
        when = result.times[-1] * j / 10
        u = interpolate(result.times, result.U, when)
        k = interpolate(result.times, result.K, when)
        e = interpolate(result.times, result.E, when)
        drift = (e - initial) / abs(initial) if initial != 0 else None
        print(f"{j / 10:>8.1f} {when / 86400:>13.5g} {u:>19.5g} {k:>19.5g} {e:>19.5g} {fmt(drift):>19}")
    if initial == 0:
        print("Fractional energy departure is undefined because the initial total energy is zero.")


def main(argv=None):
    args = parse_args(argv)
    parameters = {key: getattr(args, key) for key in DEFAULTS}
    try:
        result = integrate_binary(**parameters)
        print_summary(result, parameters)
        plot_binary(result, OUTPUT_TYPES[args.output_type])
    except (ValueError, RuntimeError, OverflowError) as error:
        raise SystemExit(f"Binary: {error}") from error


if __name__ == "__main__":
    main()
