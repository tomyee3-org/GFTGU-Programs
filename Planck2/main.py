"""Run Planck2 black-body spectra from the command line.

Inputs (use `python main.py --help` for their current defaults):

`--T K`: positive finite black-body temperature in kelvin.
`--quantity {wavelength,frequency,energy_density}`: choose spectral radiance
    per wavelength (W m^-3 sr^-1), spectral radiance per frequency
    (W m^-2 sr^-1 Hz^-1), or spectral energy density per frequency
    (J m^-3 Hz^-1), respectively. Their peak locations differ by definition.
`--n_steps N`: number of equal dimensionless x intervals (1 through 1000000);
    the grid contains N+1 endpoints.
`--x_min`, `--x_max`: lower and upper bounds of the positive dimensionless
    integration domain x = h nu / (k T).
`--x_low`, `--x_high`: approximation switches inside the domain. The
    small-x Rayleigh-Jeans form is used below x_low and the large-x Wien
    form above x_high.
`--corner {upper_right,upper_left,lower_right,lower_left}`: position of the
    plot annotation; underscores map to the plotter's spaced internal names.
`--y_frac_window FRACTION`: fraction of the peak used to window the
    displayed physical coordinate (0 to 1). Zero shows the whole domain;
    this setting does not alter integration or peak search.

Examples:
    python main.py --quantity wavelength --T 5900
    python main.py --quantity frequency --T 2.725
    python main.py --quantity energy_density --corner lower_left
    python main.py --n_steps 20000 --x_low 0.05 --x_high 20
"""

import argparse
import math

import planck2_physics
from planck2_driver import MAX_STEPS, run_planck2
from planck2_plot import plot_planck2
from planck2_physics import PlanckDomain, SHAPE_EXPONENT


CORNER_NAMES = {
    "upper_right": "upper right",
    "upper_left": "upper left",
    "lower_right": "lower right",
    "lower_left": "lower left",
}


def _finite_float(text):
    try:
        number = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite.")
    return number


def _positive_float(text):
    number = _finite_float(text)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return number


def _fraction(text):
    number = _finite_float(text)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("fraction must lie between 0 and 1.")
    return number


def _step_count(text):
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if not 1 <= number <= MAX_STEPS:
        raise argparse.ArgumentTypeError(
            f"n_steps must be between 1 and {MAX_STEPS}."
        )
    return number


def _exact_dimensionless_area(p: int) -> float:
    """Exact integral from 0 to infinity of x^p/(exp(x)-1) dx for p=3 or 5."""
    if p == 3:
        return math.pi**4 / 15.0
    if p == 5:
        return 8.0 * math.pi**6 / 63.0
    raise ValueError(f"No closed form on hand for p={p}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="Planck2",
        description="Explore black-body spectra in dimensionless and SI forms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Planck2 {planck2_physics.MODEL_VERSION} "
            f"(build {planck2_physics.BUILD_ID})"
        ),
    )
    parser.add_argument(
        "--T", type=_positive_float, default=5900.0, metavar="K",
        help="positive black-body temperature [K]",
    )
    parser.add_argument(
        "--quantity", choices=tuple(SHAPE_EXPONENT), default="wavelength",
        help=("spectral quantity: wavelength is radiance per wavelength; "
              "frequency is radiance per frequency; energy_density is "
              "energy per volume per frequency"),
    )
    parser.add_argument(
        "--n_steps", type=_step_count, default=2000, metavar="N",
        help="number of equal x intervals (1 through 1000000); N+1 samples",
    )
    parser.add_argument(
        "--x_min", type=_positive_float, default=0.01, metavar="X",
        help="positive lower limit of dimensionless integration domain",
    )
    parser.add_argument(
        "--x_max", type=_positive_float, default=100.0, metavar="X",
        help="upper limit of dimensionless domain; greater than x_min",
    )
    parser.add_argument(
        "--x_low", type=_positive_float, default=0.05, metavar="X",
        help="threshold below which the Rayleigh-Jeans approximation is used",
    )
    parser.add_argument(
        "--x_high", type=_positive_float, default=20.0, metavar="X",
        help="threshold above which the Wien approximation is used",
    )
    parser.add_argument(
        "--corner", choices=tuple(CORNER_NAMES), default="upper_right",
        help=("annotation location: upper_right, upper_left, lower_right, "
              "or lower_left"),
    )
    parser.add_argument(
        "--y_frac_window", type=_fraction, default=0.003, metavar="FRAC",
        help="display window relative to peak (0 shows the full coordinate domain)",
    )
    args = parser.parse_args(argv)
    try:
        PlanckDomain(args.x_min, args.x_max, args.x_low, args.x_high).validate()
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    domain = PlanckDomain(
        x_min=args.x_min,
        x_max=args.x_max,
        x_low=args.x_low,
        x_high=args.x_high,
    )

    try:
        result = run_planck2(
            T=args.T,
            quantity=args.quantity,
            n_steps=args.n_steps,
            domain=domain,
        )
    except (ValueError, OverflowError, RuntimeError) as exc:
        raise SystemExit(f"Planck2 input/model error: {exc}") from exc

    exact_shape = _exact_dimensionless_area(SHAPE_EXPONENT[args.quantity])
    print(
        f"Planck2 {result.model_version} (build {result.build_id}) — "
        f"peak at x = {result.x_peak:.6f}"
    )
    print(
        f"Dimensionless area = {result.dimensionless_area:.6f} "
        f"(0..infinity exact value: {exact_shape:.6f})"
    )
    print(
        f"Physical integral = {result.physical_integral:.6e} "
        f"{result.physical_integral_units}"
    )
    print(
        f"Exact 0..infinity bolometric value = {result.exact_physical_integral:.6e} "
        f"{result.physical_integral_units}"
    )
    plot_planck2(
        result,
        corner=CORNER_NAMES[args.corner],
        y_frac_window=args.y_frac_window,
    )


if __name__ == "__main__":
    main()
