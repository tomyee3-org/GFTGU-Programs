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
from numbers import Real

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.3.1"
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


def _require_finite_real(name: str, value: float) -> None:
    """Raise ValueError unless *value* is a finite, non-Boolean real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def compute_acceleration(x: float, y: float, mu: float) -> tuple[float, float]:
    """Return Newtonian gravitational acceleration components."""
    _require_finite_real("x", x)
    _require_finite_real("y", y)
    _require_finite_real("mu", mu)
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    # hypot() and the magnitude/unit-vector form avoid premature overflow in
    # x*x + y*y and r**3 for otherwise representable inputs.
    r = math.hypot(x, y)
    if r == 0.0:
        raise ValueError("The point-mass gravitational field is singular at r=0.")
    if not math.isfinite(r):
        raise ValueError("The radius is outside floating-point range.")

    try:
        acceleration_magnitude = mu / r / r
    except OverflowError as exc:
        raise ValueError(
            "The gravitational acceleration is too large for floating-point calculation."
        ) from exc
    if not math.isfinite(acceleration_magnitude):
        raise ValueError(
            "The gravitational acceleration is too large for floating-point calculation."
        )

    ax = -acceleration_magnitude * (x / r)
    ay = -acceleration_magnitude * (y / r)
    if not (math.isfinite(ax) and math.isfinite(ay)):
        raise ValueError("The acceleration components are not representable as finite numbers.")
    return ax, ay


def specific_energy(
    x: float,
    y: float,
    vx: float,
    vy: float,
    mu: float,
) -> float:
    """Return specific mechanical energy, v^2/2 - mu/r (J/kg)."""
    for name, value in (("x", x), ("y", y), ("vx", vx), ("vy", vy), ("mu", mu)):
        _require_finite_real(name, value)
    if mu <= 0.0:
        raise ValueError("mu=GM must be positive.")

    r = math.hypot(x, y)
    if r == 0.0:
        raise ValueError("Specific energy is undefined at r=0.")
    if not math.isfinite(r):
        raise ValueError("The radius is outside floating-point range.")

    speed = math.hypot(vx, vy)
    if not math.isfinite(speed):
        raise ValueError("The speed is outside floating-point range.")
    kinetic = 0.5 * speed * speed
    potential = -mu / r
    energy = kinetic + potential
    if not all(math.isfinite(value) for value in (kinetic, potential, energy)):
        raise ValueError("Specific energy is not representable as a finite number.")
    return energy


def specific_angular_momentum(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> float:
    """Return signed specific angular momentum h_z = x*vy - y*vx."""
    for name, value in (("x", x), ("y", y), ("vx", vx), ("vy", vy)):
        _require_finite_real(name, value)
    angular_momentum = x * vy - y * vx
    if not math.isfinite(angular_momentum):
        raise ValueError("Specific angular momentum is not representable as a finite number.")
    return angular_momentum
