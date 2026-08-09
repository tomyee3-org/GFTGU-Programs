"""
driver_spheregravity.py

Driver for SphereGravity.
Provides a clean interface for running the spherical-shell gravity
calculation.
"""

from physics_spheregravity import compute_acceleration_profile, OutputType


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
