"""
Find out how to launch a projectile from near Earth's surface and get it into
orbit. The program shows how sensitive the trajectory is to the initial speed.

The default "simplified" force law gives gravity a constant magnitude g,
while its direction always points toward Earth's centre.
Set force_law="inverse_square" for higher-altitude and Keplerian experiments.
"""

import argparse

import physics_earthorbit
from driver_earthorbit import run_earth_orbit, version_info
from plot_earthorbit import plot_earth_orbit


def parse_args():
    parser = argparse.ArgumentParser(
        prog="EarthOrbit",
        description=(
            "Integrate Newton's cannon thought experiment using the "
            "parameters defined in main.py."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(f"EarthOrbit {physics_earthorbit.MODEL_VERSION} "
                 f"(build {physics_earthorbit.BUILD_ID})"),
    )
    return parser.parse_args()


def main():
    parse_args()

    force_law = "simplified"

    xs, ys, xEarth, yEarth = run_earth_orbit(
        h0=300.0,
        uInit=7900.0,  # tangential/horizontal; slightly sub-orbital
        vInit=0.0,     # radial/vertical
        dt=0.4,
        maxSteps=15000,
        force_law=force_law,
    )

    metadata = version_info()
    print(
        f"EarthOrbit {metadata['model_version']} "
        f"(build {metadata['build_id']}) — {len(xs):,} trajectory samples"
    )
    plot_earth_orbit(xs, ys, xEarth, yEarth)


if __name__ == "__main__":
    main()
