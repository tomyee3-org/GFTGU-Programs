"""
Physics routines for Orbit.

State vector
------------
x, y   : position (m)
vx, vy : velocity (m/s)

Gravity
-------
For a fixed central point mass with gravitational parameter mu = GM,

    a = -mu * r_vector / r^3

The central body remains fixed, so this is a test-particle model.
"""

from __future__ import annotations

import math

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.0.0"
BUILD_ID_COVERS = (
    "physics_orbit.py",
    "driver_orbit.py",
    "main.py",
    "plot_orbit.py",
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


# IAU 2015 nominal solar mass parameter (m^3 s^-2).
GM_SUN = 1.3271244e20


def compute_acceleration(x: float, y: float, mu: float) -> tuple[float, float]:
    """Return Newtonian gravitational acceleration components."""
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(mu)):
        raise ValueError("x, y, and mu must be finite.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    r2 = x * x + y * y
    if r2 <= 0.0:
        raise ValueError("The point-mass gravitational field is singular at r=0.")

    r = math.sqrt(r2)
    factor = -mu / (r2 * r)
    return factor * x, factor * y


def specific_energy(
    x: float,
    y: float,
    vx: float,
    vy: float,
    mu: float,
) -> float:
    """Return specific mechanical energy, v^2/2 - mu/r (J/kg)."""
    if not all(math.isfinite(value) for value in (x, y, vx, vy, mu)):
        raise ValueError("Position, velocity, and mu must be finite.")
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    r = math.hypot(x, y)
    if r <= 0.0:
        raise ValueError("Specific energy is undefined at r=0.")
    return 0.5 * (vx * vx + vy * vy) - mu / r


def specific_angular_momentum(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> float:
    """Return signed specific angular momentum h_z = x*vy - y*vx."""
    if not all(math.isfinite(value) for value in (x, y, vx, vy)):
        raise ValueError("Position and velocity components must be finite.")
    return x * vy - y * vx
