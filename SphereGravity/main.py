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
    Positive finite shell mass-scale factor (default 0.001), from 1e-100 to
    1e100.  It represents nominal thickness divided by radius and scales shell
    mass and raw acceleration; it does not change the thin shell's radius or
    geometry.

``--radii R [R ...]``
    Radii at which the comparison table is printed (default 0.5 0.995 1.005
    2 3 4 5).  Each must be finite, nonnegative, at most 1e100 and different
    from the shell radius 1.  The plotted profile always uses its fixed grid from 0 to 4.995.

Examples
--------
  python main.py --nDiv 1000 --outputType acceleration
  python main.py --nDiv 1000 --outputType relative_difference --epsilon 0.002
  python main.py --radii 1.1 1.05 1.01
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
    if not physics_spheregravity.MIN_EPSILON <= value <= physics_spheregravity.MAX_EPSILON:
        raise argparse.ArgumentTypeError(
            f"value must be from {physics_spheregravity.MIN_EPSILON:g} "
            f"to {physics_spheregravity.MAX_EPSILON:g}."
        )
    return value


def _report_radius(text):
    """Parse one comparison radius: finite, nonnegative and not the shell."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("radius must be a finite nonnegative number.")
    if value > physics_spheregravity.MAX_SAMPLE_RADIUS:
        raise argparse.ArgumentTypeError(
            f"radius must not exceed {physics_spheregravity.MAX_SAMPLE_RADIUS:g}."
        )
    if value == physics_spheregravity.SHELL_RADIUS:
        raise argparse.ArgumentTypeError(
            "radius must not be the shell radius 1, where the field is discontinuous."
        )
    return value


def build_parser():
    """Return the command-line parser."""
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
            "positive shell mass-scale factor from 1e-100 to 1e100; scales mass "
            "and raw acceleration without changing shell geometry"
        ),
    )
    parser.add_argument(
        "--radii",
        type=_report_radius,
        nargs="+",
        default=list(REPORT_RADII),
        metavar="R",
        help="radii for the printed comparison table (nonnegative, at most 1e100, not 1)",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _format_value(value):
    """Format a finite result to five significant figures."""
    return f"{value:.5g}"


def _format_radius(value):
    """Format a comparison radius with enough digits to tell it from 1."""
    return f"{value:.10g}"


def print_comparison(nDiv, epsilon, radii=REPORT_RADII):
    """Print the shell mass and the shell-theorem comparison at each radius."""
    numerical_mass = physics_spheregravity.compute_shell_mass(nDiv, epsilon)
    continuum_mass = physics_spheregravity.compute_continuum_shell_mass(epsilon)
    mass_difference = (numerical_mass - continuum_mass) / continuum_mass
    radii, predicted = physics_spheregravity.compute_acceleration_at_radii(
        nDiv, radii, epsilon
    )
    estimates = physics_spheregravity.midpoint_error_estimate(nDiv, radii)
    step = math.pi / nDiv

    print("\nShell mass comparison (dimensionless units)")
    print(f"  numerical midpoint mass : {_format_value(numerical_mass)}")
    print(f"  continuum mass          : {_format_value(continuum_mass)}")
    print(f"  relative difference     : {_format_value(mass_difference)}")
    print("\nAcceleration comparison (G = 1; force for a unit test mass)")
    print(f"  latitude step h = pi/nDiv: {_format_value(step)}")
    labels = [_format_radius(radius) for radius in radii]
    width = max([len("radius")] + [len(label) for label in labels])
    print(f"  {'radius':>{width}}    predicted g       expected g        relative difference"
          "     h^2 estimate")
    coarse = False
    interior = False
    for label, radius, numerical_acceleration, estimate in zip(labels, radii, predicted, estimates):
        if radius < physics_spheregravity.SHELL_RADIUS:
            interior = True
            expected = 0.0
            residual = numerical_acceleration / continuum_mass
            comparison = f"undefined; g/M={_format_value(residual)}"
        else:
            expected = continuum_mass / radius**2
            difference = (numerical_acceleration - expected) / expected
            comparison = _format_value(difference)
        marker = ""
        scale = min(physics_spheregravity.SHELL_RADIUS,
                    abs(radius - physics_spheregravity.SHELL_RADIUS))
        if step >= 0.5 * scale:
            marker = " *"
            coarse = True
        print(
            f"  {label:>{width}}    "
            f"{_format_value(numerical_acceleration):>12}    "
            f"{_format_value(expected):>12}    {comparison:<28}"
            f"{_format_value(estimate):>12}{marker}"
        )
    if interior:
        print("  (Relative difference is undefined inside because expected g is zero.)")
    if coarse:
        print("  (* h is not small compared with |r - 1| or 1; the h^2 estimate does not apply.)")


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
    print_comparison(args.nDiv, args.epsilon, args.radii)
    plot_spheregravity(radius, accel, outputType=output_type)


if __name__ == "__main__":
    main()
