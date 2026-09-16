"""Command-line entry point for the static polytropic TOV neutron-star model.

Run ``python main.py --help`` for examples and current defaults. Parameters:
  --gamma: polytropic exponent in p = K*rho**gamma; greater than one.
  --pC: stellar central pressure [Pa]; positive and finite.
  --K: polytropic EOS constant [SI units depend on gamma]; positive and finite.
  --steps_per_scale: nominal radial resolution, at least 50 points per scale.
  --max_steps: integration safety limit, at least 100 radial steps.
  --output_type: pressure plots pressure against radius; density plots mass
      density against radius; mass plots enclosed gravitational mass against
      radius. All three quantities are calculated on every run.
  --log_y: use a logarithmic vertical axis for pressure or density. The
      mass plot always uses a linear vertical axis.

Changing gamma while keeping K fixed changes the physical equation of state;
for a controlled comparison, recalculate K for a chosen central density.
"""

import argparse
import math

import physics_neutron
from driver_neutron import MAX_NUMERICAL_CONTROL, compute_neutron_star
from plot_neutron import plot_neutron, print_model_summary


DEFAULTS = {
    "gamma": 1.666667,
    "pC": 1.26e35,
    "K": 5.3802e3,
    "steps_per_scale": 400,
    "max_steps": 200_000,
    "output_type": "pressure",
    "log_y": False,
}


def positive_finite(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("enter a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("enter a finite positive number")
    return value


def resolution(minimum):
    def convert(text):
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"enter an integer from {minimum} through {MAX_NUMERICAL_CONTROL}") from exc
        if not minimum <= value <= MAX_NUMERICAL_CONTROL:
            raise argparse.ArgumentTypeError(f"enter an integer from {minimum} through {MAX_NUMERICAL_CONTROL}")
        return value
    return convert


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="Neutron",
        description="Integrate a static, spherical neutron star using the TOV equation and a pedagogical polytropic equation of state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"Neutron {physics_neutron.MODEL_VERSION} (build {physics_neutron.BUILD_ID})")
    parser.add_argument("--gamma", type=positive_finite, default=DEFAULTS["gamma"],
                        help="polytropic exponent in p=K*rho^gamma (must exceed 1)")
    parser.add_argument("--pC", type=positive_finite, default=DEFAULTS["pC"],
                        help="central pressure [Pa]")
    parser.add_argument("--K", type=positive_finite, default=DEFAULTS["K"],
                        help="EOS constant [SI units vary with gamma]; choose with pC and central density")
    parser.add_argument("--steps_per_scale", type=resolution(50),
                        default=DEFAULTS["steps_per_scale"],
                        help="nominal number of radial steps per pressure scale, 50 to 10000000")
    parser.add_argument("--max_steps", type=resolution(100), default=DEFAULTS["max_steps"],
                        help="maximum number of outward integration steps, 100 to 10000000")
    parser.add_argument("--output_type", type=lambda text: text.strip().lower(),
                        choices=("pressure", "density", "mass"),
                        default=DEFAULTS["output_type"],
                        help="pressure: radial pressure; density: radial mass density; mass: enclosed gravitational mass")
    parser.add_argument("--log_y", action=argparse.BooleanOptionalAction,
                        default=DEFAULTS["log_y"],
                        help="logarithmic y-axis for pressure/density; mass remains linear")
    args = parser.parse_args(argv)
    if args.gamma <= 1:
        parser.error("--gamma must be greater than 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        data = compute_neutron_star(
            args.gamma, args.pC, args.K,
            steps_per_scale=args.steps_per_scale,
            max_steps=args.max_steps,
        )
    except (ValueError, RuntimeError, OverflowError) as exc:
        raise SystemExit(f"Neutron: {exc}") from exc
    print(f"Neutron {data['model_version']} (build {data['build_id']})")
    print_model_summary(data)
    plot_neutron(data, args.output_type, log_y=args.log_y)
    return data


if __name__ == "__main__":
    main()
