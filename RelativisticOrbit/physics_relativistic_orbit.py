"""
Physics for RelativisticOrbit.

The Schwarzschild mode integrates the Newtonian-looking Cartesian form of the
equatorial timelike Schwarzschild geodesic when the independent variable is the
test particle's proper time.  A Newtonian mode is also provided for direct
comparison.
"""

from __future__ import annotations

import math
from numbers import Real

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.0.1"
BUILD_ID_COVERS = (
    "physics_relativistic_orbit.py",
    "driver_relativistic_orbit.py",
    "main.py",
    "plot_relativistic_orbit.py",
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

# IAU 2015 nominal solar mass parameter and exact SI speed of light.
GM_SUN = 1.3271244e20          # m^3 s^-2
C = 299_792_458.0              # m s^-1, exact
C2 = C * C

HORIZON_RADIUS = 2.0 * GM_SUN / C2
ISCO_RADIUS = 6.0 * GM_SUN / C2
PHOTON_ORBIT_RADIUS = 3.0 * GM_SUN / C2


def _require_finite_real(name: str, value: Real) -> None:
    """Reject non-real, Boolean, NaN, and infinite numeric inputs cleanly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _model_key(model: str) -> str:
    """Return the normalized model name or raise a student-facing error."""
    if not isinstance(model, str):
        raise ValueError('model must be "schwarzschild" or "newtonian".')
    key = model.lower()
    if key not in ("schwarzschild", "newtonian"):
        raise ValueError('model must be "schwarzschild" or "newtonian".')
    return key


def orbital_constants(x_init: float, u_init: float) -> tuple[float, float]:
    """
    Return the conserved specific angular momentum h and Schutz's Q.

    The particle starts at (x_init, 0) with (dx/dtau, dy/dtau)=(0, u_init),
    where tau is the particle's proper time.

      h = x_init * u_init
      K = h/2
      Q = 12 K^2/c^2 = 3 h^2/c^2

    Q is retained because it provides a direct bridge to Schutz's notation.
    """
    _require_finite_real("x_init", x_init)
    _require_finite_real("u_init", u_init)

    h = x_init * u_init
    if not math.isfinite(h):
        raise ValueError("The initial data produce a non-finite angular momentum.")

    h2 = h * h
    if not math.isfinite(h2):
        raise ValueError("The initial data are too large to evaluate h^2 safely.")

    # Divide before multiplying by 3 to avoid an avoidable intermediate
    # overflow when h^2 is large but Q itself is still representable.
    q = 3.0 * (h2 / C2)
    if not math.isfinite(q):
        raise ValueError("The initial data produce a non-finite relativistic correction.")
    return h, q


def central_acceleration(
    x: float,
    y: float,
    h: float,
    model: str = "schwarzschild",
) -> tuple[float, float]:
    """
    Return d^2(x,y)/dtau^2 for the selected orbital model.

    schwarzschild:
        a = -GM/r^3 * (1 + 3 h^2/(c^2 r^2)) * r_vector

    newtonian:
        a = -GM/r^3 * r_vector

    In Schwarzschild mode this is an exact rewriting of the equatorial timelike
    Schwarzschild geodesic for the spatial orbit r(tau), phi(tau).  The
    "acceleration" here is a coordinate second derivative; a freely falling
    particle's physical proper acceleration is zero.
    """
    _require_finite_real("x", x)
    _require_finite_real("y", y)
    _require_finite_real("h", h)

    r2 = x * x + y * y
    if not math.isfinite(r2):
        raise ValueError("Orbit coordinates are too large to evaluate r^2 safely.")
    if r2 <= 0.0:
        raise ValueError("The test particle reached r=0, where this coordinate equation is singular.")

    r = math.sqrt(r2)
    r3 = r * r2
    if r3 <= 0.0 or not math.isfinite(r3):
        raise ValueError("Orbit coordinates are too large to evaluate r^3 safely.")

    model_key = _model_key(model)
    if model_key == "schwarzschild":
        h2 = h * h
        if not math.isfinite(h2):
            raise ValueError("Angular momentum is too large to evaluate h^2 safely.")
        correction = 1.0 + 3.0 * (h2 / C2) / r2
        if not math.isfinite(correction):
            raise ValueError("The relativistic acceleration correction is non-finite.")
    else:
        correction = 1.0

    factor = -GM_SUN * correction / r3
    ax = factor * x
    ay = factor * y
    if not (math.isfinite(ax) and math.isfinite(ay)):
        raise ValueError("The orbit acceleration is non-finite.")
    return ax, ay


def specific_angular_momentum(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> float:
    """Return h = x*dy/dtau - y*dx/dtau."""
    for name, value in (("x", x), ("y", y), ("vx", vx), ("vy", vy)):
        _require_finite_real(name, value)
    h = x * vy - y * vx
    if not math.isfinite(h):
        raise ValueError("The specific angular momentum is non-finite.")
    return h


def effective_specific_energy(
    x: float,
    y: float,
    vx: float,
    vy: float,
    h_constant: float,
    model: str = "schwarzschild",
) -> float:
    """
    Conserved energy integral for the Newtonian-like proper-time orbit equation.

    This is a numerical diagnostic for the equation being integrated, not the
    locally measured kinetic-plus-potential energy of a relativistic observer.
    """
    for name, value in (
        ("x", x),
        ("y", y),
        ("vx", vx),
        ("vy", vy),
        ("h_constant", h_constant),
    ):
        _require_finite_real(name, value)

    r = math.hypot(x, y)
    if not math.isfinite(r):
        raise ValueError("Orbit coordinates are too large to evaluate the effective energy safely.")
    if r <= 0.0:
        raise ValueError("Effective energy is undefined at r=0.")

    speed2 = vx * vx + vy * vy
    if not math.isfinite(speed2):
        raise ValueError("Velocity is too large to evaluate the effective energy safely.")
    kinetic = 0.5 * speed2
    potential = -GM_SUN / r
    if not math.isfinite(potential):
        raise ValueError("The Newtonian potential term is non-finite.")

    model_key = _model_key(model)
    if model_key == "schwarzschild":
        h2 = h_constant * h_constant
        if not math.isfinite(h2):
            raise ValueError("Angular momentum is too large to evaluate the effective energy safely.")
        relativistic_term = (GM_SUN / r) * (h2 / C2) / (r * r)
        if not math.isfinite(relativistic_term):
            raise ValueError("The relativistic energy term is non-finite.")
        potential -= relativistic_term

    energy = kinetic + potential
    if not math.isfinite(energy):
        raise ValueError("The effective energy became non-finite.")
    return energy


def circular_proper_time_speed(radius: float) -> float:
    """
    Return dy/dtau for a timelike circular Schwarzschild geodesic.

    Valid only for radius > 3 GM/c^2.  Circular timelike geodesics between
    3 GM/c^2 and 6 GM/c^2 are unstable; those above 6 GM/c^2 are stable.
    """
    _require_finite_real("radius", radius)

    denominator = radius - 3.0 * GM_SUN / C2
    if denominator <= 0.0:
        raise ValueError("No timelike circular Schwarzschild geodesic exists at or below 3GM/c^2.")
    speed2 = GM_SUN / denominator
    if not math.isfinite(speed2):
        raise ValueError("radius is too close to 3GM/c^2 for a finite result.")
    return math.sqrt(speed2)
