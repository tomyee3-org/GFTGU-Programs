"""
The program computes the gravitational acceleration produced by a thin
spherical shell of radius 1 and thickness epsilon, by dividing the shell
into nDiv × nDiv tiles and treating each tile as a point mass.

The gravitational acceleration is computed at 1000 radii from r = 0
to r = 4.995, skipping r = 1 (the shell radius). Plotting out to 5
times the shell radius gives a clear view of both the interior (r < 1)
and exterior (r > 1) regions.
"""

from typing import Literal

import numpy as np

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.2.0"
BUILD_ID_COVERS = (
    "physics_spheregravity.py",
    "driver_spheregravity.py",
    "main.py",
    "plot_spheregravity.py",
)


def _compute_build_id() -> str:
    """Return a short, reproducible identifier for the core source files.

    Files are read as UTF-8 text with universal-newline conversion, so merely
    switching between LF and CRLF line endings does not create a new build.
    Filename and byte-length framing prevents ambiguous concatenations.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            path = os.path.join(here, name)
            with open(path, "r", encoding="utf-8", newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()

OutputType = Literal["acceleration", "relative difference"]

SHELL_RADIUS = 1.0
DEFAULT_EPSILON = 0.001
NUM_RADII = 1000
RADIUS_STEP = 0.005
SURFACE_INDEX = round(SHELL_RADIUS / RADIUS_STEP)
MAX_NDIV = 100_000
MAX_VECTOR_ELEMENTS = 1_000_000


def _validate_nDiv(nDiv):
    """Validate the angular-division count used by the shell calculation."""
    if not isinstance(nDiv, (int, np.integer)) or isinstance(nDiv, bool) or nDiv <= 0:
        raise ValueError("nDiv must be a positive integer.")
    if nDiv > MAX_NDIV:
        raise ValueError(f"nDiv must not exceed {MAX_NDIV}.")


def _validate_epsilon(epsilon):
    """Validate the shell mass-scale factor."""
    if not (
        isinstance(epsilon, (int, float, np.integer, np.floating))
        and not isinstance(epsilon, (bool, np.bool_))
    ):
        raise ValueError("epsilon must be a positive finite number.")

    try:
        is_valid = np.isfinite(epsilon) and epsilon > 0
    except (TypeError, ValueError, OverflowError):
        is_valid = False

    if not is_valid:
        raise ValueError("epsilon must be a positive finite number.")


def _compute_ring_geometry(nDiv, epsilon):
    """Return vectorized ring geometry for the optimized implementation."""
    dPhi = 2.0 * np.pi / nDiv
    dTheta = 0.5 * dPhi
    theta = (np.arange(nDiv, dtype=float) + 0.5) * dTheta - 0.5 * np.pi
    sin_theta = np.sin(theta)
    ring_mass = (
        dTheta
        * dPhi
        * np.cos(theta)
        * epsilon
        * SHELL_RADIUS**2
        * nDiv
    )
    return sin_theta, ring_mass


def compute_shell_mass(nDiv, epsilon=DEFAULT_EPSILON):
    """
    Compute shell mass with the direct textbook midpoint loop.
    """
    _validate_nDiv(nDiv)
    _validate_epsilon(epsilon)

    dPhi = 2.0 * np.pi / nDiv
    dTheta = 0.5 * dPhi
    theta = 0.5 * dTheta - 0.5 * np.pi
    mass = 0.0

    for _ in range(nDiv):
        dm = (
            dTheta
            * dPhi
            * np.cos(theta)
            * epsilon
            * SHELL_RADIUS**2
        )
        mass += dm * nDiv
        theta += dTheta

    return mass


def _validate_profile_inputs(nDiv, outputType, epsilon):
    """Validate arguments shared by both profile implementations."""
    _validate_nDiv(nDiv)
    if outputType not in ("acceleration", "relative difference"):
        raise ValueError(
            "outputType must be 'acceleration' or 'relative difference'."
        )
    _validate_epsilon(epsilon)


def compute_acceleration_profile_textbook(
    nDiv,
    outputType: OutputType = "acceleration",
    epsilon=DEFAULT_EPSILON,
):
    """Compute the profile with transparent textbook-style nested loops."""
    _validate_profile_inputs(nDiv, outputType, epsilon)

    dPhi = 2.0 * np.pi / nDiv
    dTheta = 0.5 * dPhi
    mass = compute_shell_mass(nDiv, epsilon)

    radius = np.zeros(NUM_RADII)
    acceleration = np.zeros(NUM_RADII)

    # At each observation radius, add the acceleration from every latitude
    # ring. This deliberately mirrors the numerical-integration discussion in
    # the textbook rather than hiding the sum inside NumPy broadcasting.
    for j in range(NUM_RADII):
        r = j * RADIUS_STEP
        radius[j] = r

        if j == SURFACE_INDEX:
            continue

        accel = 0.0
        theta = 0.5 * dTheta - 0.5 * np.pi
        for _ in range(nDiv):
            sin_theta = np.sin(theta)
            distance = np.sqrt(
                SHELL_RADIUS**2
                + r * r
                + 2.0 * r * SHELL_RADIUS * sin_theta
            )
            dm = (
                dTheta
                * dPhi
                * np.cos(theta)
                * epsilon
                * SHELL_RADIUS**2
            )
            axial_separation = r + SHELL_RADIUS * sin_theta
            dAccel = dm * axial_separation / distance**3
            accel += dAccel * nDiv
            theta += dTheta

        acceleration[j] = accel

    if outputType == "relative difference":
        for j in range(NUM_RADII):
            if j < SURFACE_INDEX:
                acceleration[j] /= mass
            elif j > SURFACE_INDEX:
                newton = mass / radius[j] ** 2
                acceleration[j] = (acceleration[j] - newton) / newton

    return radius, acceleration


def compute_acceleration_profile_optimized(
    nDiv,
    outputType: OutputType = "acceleration",
    epsilon=DEFAULT_EPSILON,
):
    """
    Compute the same profile with bounded NumPy vectorization.

    Parameters:
        nDiv        — number of angular divisions
        outputType  — "acceleration" or "relative difference"
        epsilon     — positive finite shell mass-scale factor

    Returns:
        radius[]        — radii at which acceleration is evaluated
        acceleration[]  — computed acceleration or relative difference
    """

    _validate_profile_inputs(nDiv, outputType, epsilon)

    sin_theta, ring_mass = _compute_ring_geometry(nDiv, epsilon)
    mass = float(np.sum(ring_mass))

    radius = np.arange(NUM_RADII, dtype=float) * RADIUS_STEP
    acceleration = np.empty(NUM_RADII, dtype=float)

    # Work in bounded radial chunks. This vectorizes all latitude-ring
    # contributions while avoiding the large temporary arrays that a single
    # NUM_RADII-by-nDiv broadcast would require.
    chunk_size = max(1, min(NUM_RADII, MAX_VECTOR_ELEMENTS // nDiv))
    for first, last in ((0, SURFACE_INDEX), (SURFACE_INDEX + 1, NUM_RADII)):
        for start in range(first, last, chunk_size):
            stop = min(start + chunk_size, last)
            r = radius[start:stop, np.newaxis]
            distance_squared = (
                SHELL_RADIUS**2
                + r * r
                + 2.0 * r * SHELL_RADIUS * sin_theta[np.newaxis, :]
            )
            numerator = r + SHELL_RADIUS * sin_theta[np.newaxis, :]
            contributions = (
                ring_mass[np.newaxis, :]
                * numerator
                / (distance_squared * np.sqrt(distance_squared))
            )
            acceleration[start:stop] = np.sum(contributions, axis=1)

    # Preserve the legacy returned-data contract. The plotter replaces this
    # placeholder with NaN only in a plotting copy so the curve has a gap.
    acceleration[SURFACE_INDEX] = 0.0

    # Relative difference mode
    if outputType == "relative difference":
        interior = radius < SHELL_RADIUS
        exterior = radius > SHELL_RADIUS
        acceleration[interior] /= mass
        newton = mass / radius[exterior] ** 2
        acceleration[exterior] = (acceleration[exterior] - newton) / newton
        acceleration[SURFACE_INDEX] = 0.0

    return radius, acceleration


# IMPLEMENTATION SELECTION
#
# The transparent textbook implementation is the normal program behavior.
# To compare it with the optimized NumPy implementation, comment out the
# first assignment and uncomment the second. Both functions have identical
# signatures and return values, so driver_spheregravity.py needs no changes.
compute_acceleration_profile = compute_acceleration_profile_textbook
# compute_acceleration_profile = compute_acceleration_profile_optimized
