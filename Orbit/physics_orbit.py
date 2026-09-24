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
from typing import NamedTuple

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.5.0"
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


class KeplerianElements(NamedTuple):
    """Planar osculating elements derived from one position/velocity state."""

    classification: str
    specific_energy: float
    specific_angular_momentum: float
    eccentricity: float
    semimajor_axis: float | None
    semilatus_rectum: float
    periapsis_radius: float
    apoapsis_radius: float | None
    orbital_period: float | None
    periapsis_longitude_degrees: float | None
    initial_true_anomaly_degrees: float | None


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


def keplerian_elements(
    x: float,
    y: float,
    vx: float,
    vy: float,
    mu: float,
) -> KeplerianElements:
    """Return meaningful planar Keplerian elements for an initial state.

    Inclination and longitude of the ascending node are not represented by
    this two-dimensional model.  For a circular orbit, periapsis direction
    and true anomaly are undefined and are returned as ``None``.
    """
    energy = specific_energy(x, y, vx, vy, mu)
    angular_momentum = specific_angular_momentum(x, y, vx, vy)
    radius = math.hypot(x, y)
    speed = math.hypot(vx, vy)
    speed_squared = speed * speed
    radial_velocity_product = x * vx + y * vy

    eccentricity_x = (
        (speed_squared - mu / radius) * x
        - radial_velocity_product * vx
    ) / mu
    eccentricity_y = (
        (speed_squared - mu / radius) * y
        - radial_velocity_product * vy
    ) / mu
    eccentricity = math.hypot(eccentricity_x, eccentricity_y)
    semilatus_rectum = angular_momentum * (angular_momentum / mu)

    derived = (
        speed_squared,
        radial_velocity_product,
        eccentricity_x,
        eccentricity_y,
        eccentricity,
        semilatus_rectum,
    )
    if not all(math.isfinite(value) for value in derived):
        raise ValueError("The Keplerian elements are outside floating-point range.")

    energy_scale = max(0.5 * speed_squared, mu / radius)
    energy_tolerance = 1.0e-12 * energy_scale
    if energy < -energy_tolerance:
        classification = "elliptic"
        semimajor_axis = -mu / (2.0 * energy)
        apoapsis_radius = semimajor_axis * (1.0 + eccentricity)
        orbital_period = 2.0 * math.pi * math.sqrt(
            semimajor_axis * semimajor_axis * semimajor_axis / mu
        )
    elif energy > energy_tolerance:
        classification = "hyperbolic"
        semimajor_axis = -mu / (2.0 * energy)
        apoapsis_radius = None
        orbital_period = None
    else:
        classification = "parabolic"
        semimajor_axis = None
        apoapsis_radius = None
        orbital_period = None

    periapsis_radius = semilatus_rectum / (1.0 + eccentricity)
    if eccentricity <= 1.0e-12:
        periapsis_longitude = None
        initial_true_anomaly = None
    else:
        periapsis_longitude = math.degrees(
            math.atan2(eccentricity_y, eccentricity_x)
        ) % 360.0
        dot_product = eccentricity_x * x + eccentricity_y * y
        cross_product = eccentricity_x * y - eccentricity_y * x
        initial_true_anomaly = math.degrees(
            math.atan2(cross_product, dot_product)
        ) % 360.0

    final_values = (
        semimajor_axis,
        semilatus_rectum,
        periapsis_radius,
        apoapsis_radius,
        orbital_period,
        periapsis_longitude,
        initial_true_anomaly,
    )
    if not all(value is None or math.isfinite(value) for value in final_values):
        raise ValueError("The Keplerian elements are outside floating-point range.")

    return KeplerianElements(
        classification=classification,
        specific_energy=energy,
        specific_angular_momentum=angular_momentum,
        eccentricity=eccentricity,
        semimajor_axis=semimajor_axis,
        semilatus_rectum=semilatus_rectum,
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
        orbital_period=orbital_period,
        periapsis_longitude_degrees=periapsis_longitude,
        initial_true_anomaly_degrees=initial_true_anomaly,
    )
