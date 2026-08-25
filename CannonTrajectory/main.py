"""
Investigate the most elementary problem of motion in gravity:
the trajectory of a cannonball near the surface of the Earth.
"""

import argparse

import physics_cannon
from driver_cannon import run_cannon_trajectory, version_info
from plot_cannon import plot_cannon


def parse_args():
    parser = argparse.ArgumentParser(
        prog="CannonTrajectory",
        description=(
            "Integrate Newtonian projectile motion using the parameters "
            "defined in main.py."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(f"CannonTrajectory {physics_cannon.MODEL_VERSION} "
                 f"(build {physics_cannon.BUILD_ID})"),
    )
    return parser.parse_args()


def main():
    parse_args()

    xs, hs = run_cannon_trajectory(
        speed=100.0,
        angle_deg=45.0,
        dt=0.1,
        method="improved",
        max_steps=100_000,
    )

    metadata = version_info()
    print(
        f"CannonTrajectory {metadata['model_version']} "
        f"(build {metadata['build_id']}) — {len(xs):,} trajectory samples"
    )
    plot_cannon(xs, hs)


if __name__ == "__main__":
    main()
