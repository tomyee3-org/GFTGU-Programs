"""
Test Newton's theorem that the gravitational attraction outside
a sphere is the same as if all its mass were concentrated at its center.
"""

import argparse

import physics_spheregravity
from driver_spheregravity import get_version_info, run_spheregravity
from plot_spheregravity import plot_spheregravity
from physics_spheregravity import OutputType


def parse_args():
    parser = argparse.ArgumentParser(
        prog="SphereGravity",
        description="Test Newton's gravitational theorem for a spherical shell.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"SphereGravity {physics_spheregravity.MODEL_VERSION} "
            f"(build {physics_spheregravity.BUILD_ID})"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()


# User-adjustable settings
nDiv = 100
outputType: OutputType = "relative difference"

version_info = get_version_info()
print(
    f"SphereGravity {version_info['model_version']} "
    f"(build {version_info['build_id']})"
)

radius, accel = run_spheregravity(nDiv=nDiv, outputType=outputType)
plot_spheregravity(radius, accel, outputType=outputType)
