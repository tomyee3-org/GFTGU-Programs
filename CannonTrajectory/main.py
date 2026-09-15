"""Command-line entry point for CannonTrajectory.

The program integrates the trajectory of a projectile near Earth's surface.
Its input parameters are:

``speed``
    Positive launch speed in metres per second.
``angle_deg``
    Launch angle in degrees from the horizontal, from 0 through 90.
``dt``
    Positive fixed integration timestep in seconds.
``max_steps``
    Safety limit on stored trajectory points, including the launch point.
``method``
    Integration selector: ``euler`` uses first-order forward Euler;
    ``improved`` uses second-order improved Euler (Heun's method).

Examples
--------
  # Default 100 m/s launch at 45 degrees using improved Euler
  python main.py

  # Compare forward Euler at a coarser timestep
  python main.py --method euler --dt 0.5

  # Change the launch conditions
  python main.py --speed 150 --angle_deg 30
"""

import argparse

import physics_cannon
from driver_cannon import (
    interpolated_landing_range,
    interpolated_flight_time,
    interpolated_maximum_height,
    run_cannon_trajectory,
    version_info,
)
from plot_cannon import plot_cannon


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CannonTrajectory",
        description=(
            "Integrate Newtonian projectile motion near Earth's surface "
            "with constant gravity and no air resistance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(f"CannonTrajectory {physics_cannon.MODEL_VERSION} "
                 f"(build {physics_cannon.BUILD_ID})"),
    )
    parser.add_argument(
        "--speed", type=float, default=100.0, metavar="M_PER_S",
        help="positive launch-speed magnitude [m/s]",
    )
    parser.add_argument(
        "--angle_deg", type=float, default=45.0, metavar="DEGREES",
        help="launch angle above the horizontal, from 0 through 90 degrees",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, metavar="SECONDS",
        help="positive fixed integration timestep [s]",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100_000, metavar="N",
        help="maximum stored points, including launch; at least 2",
    )
    parser.add_argument(
        "--method", choices=("euler", "improved"), default="improved",
        help=(
            "integration method: euler is first-order forward Euler; "
            "improved is second-order improved Euler (Heun)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        xs, hs = run_cannon_trajectory(
            speed=args.speed,
            angle_deg=args.angle_deg,
            dt=args.dt,
            method=args.method,
            max_steps=args.max_steps,
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as exc:
        raise SystemExit(f"CannonTrajectory: {exc}") from exc

    metadata = version_info()
    range_m = interpolated_landing_range(xs, hs)
    height_m = interpolated_maximum_height(hs)
    flight_time_s = interpolated_flight_time(hs, args.dt)
    print(
        f"CannonTrajectory {metadata['model_version']} "
        f"(build {metadata['build_id']}) — {len(xs):,} trajectory samples"
    )
    print(f"Range (interpolated ground crossing): {range_m:.5g} m")
    print(f"Maximum height (parabolic interpolation): {height_m:.5g} m")
    print(f"Flight time (interpolated ground crossing): {flight_time_s:.5g} s")
    plot_cannon(xs, hs)


if __name__ == "__main__":
    main()
