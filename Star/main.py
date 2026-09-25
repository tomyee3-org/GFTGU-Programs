"""Command-line entry point for Star.

Star integrates a simple Newtonian polytrope from the center to its
zero-pressure surface. Every ordinary model and plotting input is available at
the command line:

``--p_c PA``
    Positive finite central pressure in pascals (default 7.158e15).

``--T_c K``
    Positive finite central temperature in kelvin (default 2.263e7).

``--mu VALUE``
    Positive finite effective mean molecular weight in proton masses (default
    1.285). This is a simplified model parameter, not a detailed composition.

``--gamma VALUE``
    Polytropic exponent (default 1.36). It must exceed 1.2 so the polytrope has
    an ordinary finite-radius zero-pressure surface.

``--max_points N``
    Grid-point capacity for each integration attempt (default 2000, from 3 to
    1000000). A model that needs more points is restarted with a doubled
    radial step, so raising it refines a run only by removing restarts.

``--steps_per_scale N``
    Positive number of Euler steps per characteristic radial scale (default
    400). Larger values give finer resolution when ``max_points`` prevents a
    restart.

``--output_type {pressure,density,temperature,mass}``
    Quantity plotted against radius: pressure in Pa, density in kg/m^3,
    temperature in K, or enclosed mass in kg (default ``pressure``).

``--log_y``
    Use a logarithmic vertical axis. It is useful for pressure, density, or
    temperature, but is invalid for mass because enclosed mass is zero at the
    center. Omit this switch for a linear axis.

Each run also prints the surface radius and total mass; the radial step, the
number of grid points and the number of step doublings (restarts); the
polytropic index and the radius and mass of the Lane-Emden solution of the
same polytrope, computed as a numerical reference, with the relative
difference of the integrated values and a warning if that reference is
uncertain in its seventh significant figure;
and linearly interpolated pressure, density, temperature, and enclosed mass
at 0%, 25%, 50%, 75%, and 90% of the surface radius. Numerical results are
printed to five significant figures.

Examples
--------
  python main.py
  python main.py --output_type temperature
  python main.py --output_type density --log_y
  python main.py --T_c 2.4893e7 --steps_per_scale 800 --max_points 4000
"""

import argparse
import math

import physics_star
from driver_star import (
    MAX_POINTS_LIMIT,
    OUTPUT_TYPES,
    integrate_star,
    interpolate_profile_checkpoints,
)
from plot_star import plot_star_structure


def _positive_float(text):
    """Parse a positive finite command-line number."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number.")
    return value


def _finite_radius_gamma(text):
    """Parse a finite polytropic exponent greater than 6/5."""
    value = _positive_float(text)
    if value <= 1.2:
        raise argparse.ArgumentTypeError(
            "gamma must exceed 1.2 for a finite-radius polytrope."
        )
    return value


def _integer_at_least(minimum, maximum=None):
    """Return an argparse converter for integers at or above minimum.

    When maximum is given, larger integers are rejected as well.
    """
    def converter(text):
        try:
            value = int(text)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"value must be an integer of at least {minimum}."
            )
        if maximum is not None and value > maximum:
            raise argparse.ArgumentTypeError(
                f"value must be an integer of at most {maximum}."
            )
        return value

    return converter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="Star",
        description="Compute and plot a simple Newtonian polytropic star.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Star {physics_star.MODEL_VERSION} "
            f"(build {physics_star.BUILD_ID})"
        ),
    )
    parser.add_argument(
        "--p_c",
        type=_positive_float,
        default=7.158e15,
        metavar="PA",
        help="positive finite central pressure [Pa]",
    )
    parser.add_argument(
        "--T_c",
        type=_positive_float,
        default=2.263e7,
        metavar="K",
        help="positive finite central temperature [K]",
    )
    parser.add_argument(
        "--mu",
        type=_positive_float,
        default=1.285,
        metavar="VALUE",
        help="positive effective mean molecular weight in proton masses",
    )
    parser.add_argument(
        "--gamma",
        type=_finite_radius_gamma,
        default=1.36,
        metavar="VALUE",
        help="polytropic exponent; must exceed 1.2 for a finite-radius model",
    )
    parser.add_argument(
        "--max_points",
        type=_integer_at_least(3, MAX_POINTS_LIMIT),
        default=2000,
        metavar="N",
        help=(
            "grid-point capacity per attempt; a model needing more points "
            "restarts with a doubled step"
        ),
    )
    parser.add_argument(
        "--steps_per_scale",
        type=_integer_at_least(1),
        default=400,
        metavar="N",
        help="Euler steps per characteristic radial scale",
    )
    parser.add_argument(
        "--output_type",
        choices=OUTPUT_TYPES,
        default="pressure",
        metavar="QUANTITY",
        help=(
            "quantity to plot: pressure [Pa], density [kg/m^3], temperature "
            "[K], or enclosed mass [kg]"
        ),
    )
    parser.add_argument(
        "--log_y",
        action="store_true",
        help=(
            "use a logarithmic vertical axis for pressure, density, or "
            "temperature; invalid with mass"
        ),
    )
    return parser.parse_args(argv)


def _five_significant(value):
    """Format one finite result with five significant digits."""
    return f"{value:.5g}"


def _print_numerical_grid(result):
    """Print the radial step actually used and how it was reached."""
    print("\n  Numerical grid")
    print(f"  Radial step (m): {_five_significant(result.radial_step)}")
    print(f"  Grid points, centre to surface: {len(result.radius)}")
    print(f"  Restarts (radial step doubled): {result.restart_count}")


def _print_lane_emden_comparison(result):
    """Print the Lane-Emden reference radius and mass beside the integrated ones."""
    print("\n  Lane-Emden solution of the same polytrope (numerical reference)")
    try:
        reference = physics_star.lane_emden_solution(
            result.pressure[0], result.density[0], result.gamma
        )
    except (ValueError, OverflowError) as exc:
        print(f"  not computed: {exc}")
        return
    radius_difference = result.radius[-1] / reference.radius - 1.0
    mass_difference = result.mass[-1] / reference.mass - 1.0
    index = _five_significant(reference.n)
    if float(index) >= 5.0:
        # n is below 5 for every accepted gamma; show enough figures to say so.
        index = f"{reference.n:.12g}"
    print(f"  Polytropic index n = 1/(gamma - 1): {index}")
    print(
        f"  Lane-Emden radius (m):     {_five_significant(reference.radius)}   "
        f"integrated radius differs by {_five_significant(radius_difference)}"
    )
    print(
        f"  Lane-Emden mass (kg):      {_five_significant(reference.mass)}   "
        f"integrated mass differs by {_five_significant(mass_difference)}"
    )
    if reference.relative_uncertainty > physics_star.LANE_EMDEN_UNCERTAINTY_WARNING:
        print(
            "  Warning: gamma is so close to 6/5 that these Lane-Emden values "
            f"are uncertain by about {reference.relative_uncertainty:.1g} (relative)."
        )


def print_structure_summary(result):
    """Print global and interpolated stellar properties to five figures."""
    print("\nStellar structure summary")
    print(f"  Radius (m):     {_five_significant(result.radius[-1])}")
    print(f"  Total mass (kg): {_five_significant(result.mass[-1])}")
    if hasattr(result, "radial_step") and hasattr(result, "restart_count"):
        _print_numerical_grid(result)
    if getattr(result, "gamma", None) is not None:
        _print_lane_emden_comparison(result)
    print("\n  Interior checkpoints (linear interpolation in radius)")
    print(
        "  r/R     radius (m)    pressure (Pa)  density (kg/m^3)  "
        "temperature (K)  mass within r (kg)"
    )
    for sample in interpolate_profile_checkpoints(result):
        print(
            f"  {sample.radius_fraction:>4.0%}  "
            f"{_five_significant(sample.radius):>13}  "
            f"{_five_significant(sample.pressure):>13}  "
            f"{_five_significant(sample.density):>16}  "
            f"{_five_significant(sample.temperature):>15}  "
            f"{_five_significant(sample.mass):>18}"
        )


def main(argv=None):
    args = parse_args(argv)
    if args.output_type == "mass" and args.log_y:
        raise SystemExit(
            "Star input/plot error: --log_y cannot be used with "
            "--output_type mass because enclosed mass is zero at the center."
        )

    try:
        result = integrate_star(
            p_c=args.p_c,
            T_c=args.T_c,
            mu=args.mu,
            gamma=args.gamma,
            max_points=args.max_points,
            steps_per_scale=args.steps_per_scale,
            output_type=args.output_type,
        )
        print(f"Star {result.model_version} (build {result.build_id})")
        print_structure_summary(result)
        plot_star_structure(result, log_y=args.log_y)
    except (ValueError, OverflowError, RuntimeError) as exc:
        raise SystemExit(f"Star input/model error: {exc}") from exc


if __name__ == "__main__":
    main()
