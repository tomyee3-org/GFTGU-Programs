"""
driver_spheregravity.py

Driver for SphereGravity.
Provides a clean interface for running the spherical-shell gravity
calculation.
"""

import physics_spheregravity as phys
from physics_spheregravity import compute_acceleration_profile, OutputType


def get_version_info() -> dict[str, str]:
    """Return machine-readable model version and source build identifiers."""
    return {
        "model_version": phys.MODEL_VERSION,
        "build_id": phys.BUILD_ID,
    }


def run_spheregravity(nDiv=100, outputType: OutputType = "acceleration"):
    """
    Run the SphereGravity simulation.

    Parameters:
        nDiv        — number of angular divisions
        outputType  — "acceleration" or "relative difference"

    Returns:
        radius[], acceleration[]
    """
    return compute_acceleration_profile(nDiv, outputType)
