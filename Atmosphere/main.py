"""Command-line entry point for Atmosphere.

The program integrates hydrostatic equilibrium through a supplied temperature
profile. Every user-facing model input can be set at the command line:

``--planet_name NAME``
    Nonempty label used in the plot and printed heading (default ``Earth``).

``--g_accel VALUE``
    Positive, finite, constant gravitational acceleration in m/s^2 (default
    9.81). Larger gravity makes pressure fall more rapidly with altitude.

``--mu VALUE``
    Positive, finite mean molecular weight in proton masses (default 28.97 for
    dry air). Larger values also reduce the atmospheric scale height.

``--p0 VALUE``
    Positive, finite pressure at altitude zero in pascals (default 101300).

``--h_points LIST`` and ``--T_points LIST``
    Comma-separated altitude checkpoints in meters and their temperatures in
    kelvin. The lists must have equal lengths; altitudes must strictly increase,
    and temperatures must be positive. Between checkpoints temperature is
    linearly interpolated. Defaults form the documented hybrid Earth profile.

``--output_type {pressure,density,temperature}``
    Select the plotted quantity: ``pressure`` plots hydrostatic pressure in Pa;
    ``density`` plots ideal-gas mass density in kg/m^3; ``temperature`` plots
    the prescribed/interpolated temperature in K. Lowercase selector values
    are mapped to the driver's legacy capitalized values.

Examples
--------
  python main.py --output_type temperature
  python main.py --planet_name Mars --g_accel 3.71 --mu 44 --p0 610 \
      --h_points 0,10000,20000 --T_points 210,180,160 --output_type pressure
"""

import argparse
import math

import physics_atmosphere
from driver_atmosphere import (
    AtmosphereModel,
    AtmosphereParameters,
    extract_checkpoints,
    extract_output,
)
from plot_atmosphere import plot_atmosphere


DEFAULT_H_POINTS = (
    0.0,
    11_019.0,
    20_063.0,
    32_162.0,
    47_350.0,
    51_412.0,
    71_802.0,
    86_000.0,
    100_000.0,
    150_000.0,
    200_000.0,
    250_000.0,
    300_000.0,
    400_000.0,
    500_000.0,
)
DEFAULT_T_POINTS = (
    288.15,
    216.65,
    216.65,
    228.65,
    270.65,
    270.65,
    214.65,
    186.946,
    190.0,
    800.0,
    1080.0,
    1190.0,
    1225.0,
    1240.0,
    1240.0,
)
OUTPUT_TYPES = {
    "pressure": "Pressure",
    "density": "Density",
    "temperature": "Temperature",
}


def _nonempty_text(text):
    """Parse a nonempty command-line label."""
    value = text.strip()
    if not value:
        raise argparse.ArgumentTypeError("value must contain non-whitespace characters.")
    return value


def _positive_float(text):
    """Parse a positive finite command-line number."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number.")
    return value


def _float_list(text):
    """Parse a nonempty comma-separated list of finite numbers."""
    pieces = text.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated list without empty entries."
        )
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "every comma-separated entry must be a number."
        ) from exc
    if any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("every list entry must be finite.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="Atmosphere",
        description=(
            "Integrate a constant-gravity, constant-composition atmosphere "
            "through a piecewise-linear temperature profile."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Atmosphere {physics_atmosphere.MODEL_VERSION} "
            f"(build {physics_atmosphere.BUILD_ID})"
        ),
    )
    parser.add_argument(
        "--planet_name",
        type=_nonempty_text,
        default="Earth",
        metavar="NAME",
        help="nonempty planet or atmosphere label used in output",
    )
    parser.add_argument(
        "--g_accel",
        type=_positive_float,
        default=9.81,
        metavar="M_S2",
        help="constant positive gravitational acceleration [m/s^2]",
    )
    parser.add_argument(
        "--mu",
        type=_positive_float,
        default=28.97,
        metavar="PROTON_MASSES",
        help="positive mean molecular weight in proton masses",
    )
    parser.add_argument(
        "--p0",
        type=_positive_float,
        default=1.013e5,
        metavar="PA",
        help="positive pressure at altitude zero [Pa]",
    )
    parser.add_argument(
        "--h_points",
        type=_float_list,
        default=list(DEFAULT_H_POINTS),
        metavar="LIST",
        help="comma-separated, strictly increasing altitude checkpoints [m]",
    )
    parser.add_argument(
        "--T_points",
        type=_float_list,
        default=list(DEFAULT_T_POINTS),
        metavar="LIST",
        help="comma-separated positive checkpoint temperatures [K]",
    )
    parser.add_argument(
        "--output_type",
        choices=tuple(OUTPUT_TYPES),
        default="pressure",
        metavar="MODE",
        help=(
            "quantity to plot: pressure [Pa], density [kg/m^3], or "
            "temperature [K]"
        ),
    )
    return parser.parse_args(argv)


def _format_value(value):
    """Format a finite numerical result to five significant figures."""
    return f"{value:.5g}"


def _format_altitude(value):
    """Keep checkpoint altitudes readable while retaining useful precision."""
    return f"{value:.8g}"


def print_checkpoints(result, h_points, T_points):
    """Print pressure, p/T, and temperature at every supplied checkpoint."""
    rows = extract_checkpoints(result, h_points, T_points)
    print("\nAtmospheric checkpoints")
    print("  altitude (m)      pressure (Pa)          p/T (Pa/K)    temperature (K)")
    for row in rows:
        if row.pressure is None:
            pressure = "unavailable"
            pressure_over_temperature = "unavailable"
        else:
            pressure = _format_value(row.pressure)
            pressure_over_temperature = _format_value(
                row.pressure_over_temperature
            )
        print(
            f"  {_format_altitude(row.altitude):>12}    "
            f"{pressure:>15}    {pressure_over_temperature:>16}    "
            f"{_format_value(row.temperature):>15}"
        )
    print("  (Unavailable means the checkpoint is outside the stored positive-pressure domain.)")


def main(argv=None):
    args = parse_args(argv)
    params = AtmosphereParameters(
        planet_name=args.planet_name,
        g_accel=args.g_accel,
        mu=args.mu,
        p0=args.p0,
        h_points=args.h_points,
        T_points=args.T_points,
        output_type=OUTPUT_TYPES[args.output_type],
    )

    try:
        result = AtmosphereModel(params).run()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Atmosphere input/model error: {exc}") from exc
    print(
        f"Atmosphere {result.model_version} (build {result.build_id}) — "
        f"{result.planet_name}: {result.output_type}"
    )
    print_checkpoints(result, args.h_points, args.T_points)
    plot_atmosphere(extract_output(result))


if __name__ == "__main__":
    main()
