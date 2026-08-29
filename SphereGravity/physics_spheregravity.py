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
MODEL_VERSION = "1.0.1"
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


def _validate_nDiv(nDiv):
    """Validate the angular-division count used by the shell calculation."""
    if not isinstance(nDiv, (int, np.integer)) or isinstance(nDiv, bool) or nDiv <= 0:
        raise ValueError("nDiv must be a positive integer.")


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


def compute_shell_mass(nDiv, epsilon=DEFAULT_EPSILON):
    """
    Compute total mass of the spherical shell by summing tile masses.
    """
    _validate_nDiv(nDiv)
    _validate_epsilon(epsilon)

    degToRad = np.pi / 180.0
    dPhi = 360.0 * degToRad / nDiv
    dTheta = 0.5 * dPhi

    theta = 0.5 * dTheta - 90 * degToRad
    mass = 0.0

    for _ in range(nDiv):
        dm = dTheta * dPhi * np.cos(theta) * epsilon
        mass += dm * nDiv
        theta += dTheta

    return mass


def compute_acceleration_profile(
    nDiv,
    outputType: OutputType = "acceleration",
    epsilon=DEFAULT_EPSILON,
):
    """
    Compute gravitational acceleration at 1000 radii from r = 0 to r = 4.995
    (step 0.005), spanning 5 times the shell radius for a clear view of
    both interior and exterior behaviour.

    Parameters:
        nDiv        — number of angular divisions
        outputType  — "acceleration" or "relative difference"
        epsilon     — shell thickness

    Returns:
        radius[]        — radii at which acceleration is evaluated
        acceleration[]  — computed acceleration or relative difference
    """

    _validate_nDiv(nDiv)
    if outputType not in ("acceleration", "relative difference"):
        raise ValueError(
            "outputType must be 'acceleration' or 'relative difference'."
        )
    _validate_epsilon(epsilon)

    degToRad = np.pi / 180.0
    dPhi = 360.0 * degToRad / nDiv
    dTheta = 0.5 * dPhi

    # Precompute shell mass
    mass = compute_shell_mass(nDiv, epsilon)

    radius = np.zeros(NUM_RADII)
    acceleration = np.zeros(NUM_RADII)

    # Loop over radii: r = j * RADIUS_STEP, from 0 through 4.995.
    # The shell lies at r = SHELL_RADIUS, at SURFACE_INDEX.
    for j in range(NUM_RADII):
        r = j * RADIUS_STEP
        radius[j] = r

        if j == SURFACE_INDEX:  # r = 1 — omit the idealized shell surface
            acceleration[j] = 0.0
            continue

        accel = 0.0
        theta = 0.5 * dTheta - 90 * degToRad

        # Loop over latitude rings
        for _ in range(nDiv):
            s = np.sin(theta)
            dist = np.sqrt(1 + r*r + 2*r*s)
            dm = dTheta * dPhi * np.cos(theta) * epsilon
            x = r + s
            dAccel = dm * x / (dist**3)
            accel += dAccel * nDiv
            theta += dTheta

        acceleration[j] = accel

    # Relative difference mode
    if outputType == "relative difference":
        # Indices 0..200 inclusive (r = 0.00 through r = 1.00, i.e. inside
        # the shell and the shell itself) are normalised by dividing by
        # `mass` — which equals the Newtonian acceleration just outside the
        # shell, since Newton's g = mass/r^2 evaluates to exactly `mass` at
        # r = 1. Indices 201+ (strictly outside) use the standard relative-
        # difference formula against the Newtonian prediction at that radius.
        # Using j (not a floating-point r < 1.0 comparison) avoids any
        # ambiguity from r = j*0.005 not landing exactly on 1.0 in floating
        # point at j = 200.
        for j in range(NUM_RADII):
            if j <= SURFACE_INDEX:
                acceleration[j] = acceleration[j] / mass
            else:
                r = radius[j]
                newton = mass / (r * r)
                acceleration[j] = (acceleration[j] - newton) / newton

    return radius, acceleration
