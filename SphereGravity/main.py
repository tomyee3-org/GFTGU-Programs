"""Command-line entry point for SphereGravity.

The program tests Newton's shell theorem by replacing a thin spherical shell
with a midpoint-rule angular tiling.  Every simulation input accepted by the
driver can be supplied on the command line:

``--nDiv N``
    Positive number of latitude and azimuth divisions (default 100; maximum
    100000).  The conceptual shell contains N by N tiles.  Larger values
    improve angular resolution but take longer.

``--outputType {acceleration,relative_difference}``
    Select the plotted quantity. ``acceleration`` plots the raw numerical
    gravitational acceleration. ``relative_difference`` plots the signed
    exterior fractional error and the mass-normalized residual inside the
    shell, where the exact acceleration is zero.  The underscore makes the
    second selector safe to pass as one command-line token; internally it maps
    to the legacy value ``"relative difference"``.

``--epsilon VALUE``
    Positive finite shell mass-scale factor (default 0.001).  It represents
    nominal thickness divided by radius and scales shell mass and raw
    acceleration; it does not change the thin shell's radius or geometry.

Examples
--------
  python main.py --nDiv 1000 --outputType acceleration
  python main.py --nDiv 1000 --outputType relative_difference --epsilon 0.002
"""

import argparse

import math

import physics_spheregravity
from driver_spheregravity import get_version_info, run_spheregravity
from plot_spheregravity import plot_spheregravity


CLI_OUTPUT_TYPES = {
    "acceleration": "acceleration",
    "relative_difference": "relative difference",
}
DEFAULT_NDIV = 100
DEFAULT_OUTPUT_TYPE = "relative_difference"
REPORT_RADII = (0.5, 0.995, 1.005, 2.0, 3.0, 4.0, 5.0)


def _positive_int(text):
    """Parse a positive integer no greater than the physics safety limit."""
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    if value > physics_spheregravity.MAX_NDIV:
        raise argparse.ArgumentTypeError(
            f"value must not exceed {physics_spheregravity.MAX_NDIV}."
        )
    return value


def _positive_float(text):
    """Parse a positive finite floating-point value."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="SphereGravity",
        description="Test Newton's gravitational theorem for a spherical shell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"SphereGravity {physics_spheregravity.MODEL_VERSION} "
            f"(build {physics_spheregravity.BUILD_ID})"
        ),
    )
    parser.add_argument(
        "--nDiv",
        type=_positive_int,
        default=DEFAULT_NDIV,
        metavar="N",
        help=(
            "angular divisions in each coordinate; larger N improves resolution "
            f"but takes longer (maximum {physics_spheregravity.MAX_NDIV})"
        ),
    )
    parser.add_argument(
        "--outputType",
        choices=tuple(CLI_OUTPUT_TYPES),
        default=DEFAULT_OUTPUT_TYPE,
        help=(
            "plot acceleration (raw field), or relative_difference (exterior "
            "fractional error and interior mass-normalized residual)"
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=_positive_float,
        default=physics_spheregravity.DEFAULT_EPSILON,
        metavar="VALUE",
        help=(
            "positive shell mass-scale factor; scales mass and raw acceleration "
            "without changing shell geometry"
        ),
    )
    return parser.parse_args(argv)


def _format_value(value):
    """Format a finite result to five significant figures."""
    return f"{value:.5g}"


def print_comparison(nDiv, epsilon):
    """Print shell mass and requested shell-theorem comparison radii."""
    numerical_mass = physics_spheregravity.compute_shell_mass(nDiv, epsilon)
    continuum_mass = physics_spheregravity.compute_continuum_shell_mass(epsilon)
    mass_difference = (numerical_mass - continuum_mass) / continuum_mass
    radii, predicted = physics_spheregravity.compute_acceleration_at_radii(
        nDiv, REPORT_RADII, epsilon
    )

    print("\nShell mass comparison (dimensionless units)")
    print(f"  numerical midpoint mass : {_format_value(numerical_mass)}")
    print(f"  continuum mass          : {_format_value(continuum_mass)}")
    print(f"  relative difference     : {_format_value(mass_difference)}")
    print("\nAcceleration comparison (G = 1; force for a unit test mass)")
    print("  radius    predicted g       expected g        relative difference")
    for radius, numerical_acceleration in zip(radii, predicted):
        if radius < physics_spheregravity.SHELL_RADIUS:
            expected = 0.0
            residual = numerical_acceleration / continuum_mass
            comparison = f"undefined; g/M={_format_value(residual)}"
        else:
            expected = continuum_mass / radius**2
            difference = (numerical_acceleration - expected) / expected
            comparison = _format_value(difference)
        print(
            f"  {_format_value(radius):>6}    "
            f"{_format_value(numerical_acceleration):>12}    "
            f"{_format_value(expected):>12}    {comparison}"
        )
    print("  (Relative difference is undefined inside because expected g is zero.)")


def main(argv=None):
    """Run the configured calculation and display its plot."""
    args = parse_args(argv)
    output_type = CLI_OUTPUT_TYPES[args.outputType]

    version_info = get_version_info()
    print(
        f"SphereGravity {version_info['model_version']} "
        f"(build {version_info['build_id']})"
    )

    radius, accel = run_spheregravity(
        nDiv=args.nDiv,
        outputType=output_type,
        epsilon=args.epsilon,
    )
    print_comparison(args.nDiv, args.epsilon)
    plot_spheregravity(radius, accel, outputType=output_type)


if __name__ == "__main__":
    main()
