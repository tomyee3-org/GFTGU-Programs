"""Command-line entry point for EarthOrbit.

The program integrates Newton's cannon thought experiment.  Its user input
parameters are:

``h0``
    Initial altitude above the reference Earth surface, in metres.
``uInit``
    Initial tangential/horizontal velocity, in metres per second.
``vInit``
    Initial radial/vertical velocity, in metres per second; positive is away
    from Earth's centre at launch.
``dt``
    Positive fixed integration timestep, in seconds.
``maxSteps``
    Safety limit on stored trajectory points, including the launch point.
``force_law``
    Gravity selector: ``simplified`` keeps acceleration magnitude fixed at
    9.8 m/s² while pointing toward Earth's centre; ``inverse_square`` uses
    the Keplerian MU_EARTH/r² law and is required for meaningful orbital
    elements and escape trajectories.

The driver's ``return_diagnostics`` argument is an API return-format switch,
not a physical setting.  This command-line program enables it internally to
calculate the printed trajectory summary.

Examples
--------
  # Default, slightly sub-orbital near-surface launch
  python main.py

  # A 300 km circular inverse-square orbit (approximately)
  python main.py --h0 300000 --uInit 7726 --force_law inverse_square \
    --dt 1 --maxSteps 7000

  # An inverse-square escape trajectory
  python main.py --h0 300000 --uInit 11500 --force_law inverse_square \
    --dt 2 --maxSteps 3001
"""

import argparse

import physics_earthorbit
from driver_earthorbit import analyze_earth_orbit, run_earth_orbit, version_info
from plot_earthorbit import plot_earth_orbit


def parse_args():
    parser = argparse.ArgumentParser(
        prog="EarthOrbit",
        description=(
            "Integrate Newton's cannon thought experiment with either a "
            "constant-magnitude near-surface gravity model or inverse-square "
            "gravity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(f"EarthOrbit {physics_earthorbit.MODEL_VERSION} "
                 f"(build {physics_earthorbit.BUILD_ID})"),
    )
    parser.add_argument(
        "--h0", type=float, default=300.0, metavar="METRES",
        help="initial altitude above the reference Earth surface [m]",
    )
    parser.add_argument(
        "--uInit", type=float, default=7900.0, metavar="M_PER_S",
        help="initial tangential/horizontal velocity [m/s]",
    )
    parser.add_argument(
        "--vInit", type=float, default=0.0, metavar="M_PER_S",
        help="initial radial/vertical velocity; positive is outward [m/s]",
    )
    parser.add_argument(
        "--dt", type=float, default=0.4, metavar="SECONDS",
        help="positive fixed integration timestep [s]",
    )
    parser.add_argument(
        "--maxSteps", type=int, default=15_000, metavar="N",
        help="maximum stored trajectory points, including launch; at least 2",
    )
    parser.add_argument(
        "--force_law", choices=("simplified", "inverse_square"),
        default="simplified",
        help=(
            "gravity model: simplified keeps |g| = 9.8 m/s^2; "
            "inverse_square uses MU_EARTH/r^2 for Keplerian motion"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        result = run_earth_orbit(
            h0=args.h0,
            uInit=args.uInit,
            vInit=args.vInit,
            dt=args.dt,
            maxSteps=args.maxSteps,
            force_law=args.force_law,
            return_diagnostics=True,
        )
        xs, ys, xEarth, yEarth, ts, us, vs = result
        summary = analyze_earth_orbit(
            xs, ys, ts, us, vs,
            force_law=args.force_law,
            max_steps=args.maxSteps,
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        raise SystemExit(f"EarthOrbit: {exc}") from exc

    metadata = version_info()
    print(
        f"EarthOrbit {metadata['model_version']} "
        f"(build {metadata['build_id']}) — {len(xs):,} trajectory samples"
    )
    for line in summary["lines"]:
        print(line)
    plot_earth_orbit(xs, ys, xEarth, yEarth)


if __name__ == "__main__":
    main()
