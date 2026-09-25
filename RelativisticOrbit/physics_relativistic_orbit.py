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
from typing import NamedTuple, Optional

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.2.0"
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
    Filename and byte-length framing prevents ambiguous concatenations.  A
    physics-only copy without the complete core is labeled "unpackaged".
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
        # A physics-only copy remains importable for exploration, but it must
        # not masquerade as a traceable release build without all four files.
        return "unpackaged"


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


def circular_proper_time_speed(radius: float, model: str = "schwarzschild") -> float:
    """
    Return dy/dtau for a circular orbit of the selected model.

    schwarzschild: sqrt(GM/(r - 3GM/c^2)), the timelike circular geodesic.
    Valid only for radius > 3 GM/c^2.  Circular timelike geodesics between
    3 GM/c^2 and 6 GM/c^2 are unstable; those above 6 GM/c^2 are stable.

    newtonian: sqrt(GM/r), for any positive radius.
    """
    _require_finite_real("radius", radius)
    if _model_key(model) == "newtonian":
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        speed2 = GM_SUN / radius
        if not math.isfinite(speed2):
            raise ValueError("radius is too small for a finite circular speed.")
        return math.sqrt(speed2)

    denominator = radius - 3.0 * GM_SUN / C2
    if denominator <= 0.0:
        raise ValueError("No timelike circular Schwarzschild geodesic exists at or below 3GM/c^2.")
    speed2 = GM_SUN / denominator
    if not math.isfinite(speed2):
        raise ValueError("radius is too close to 3GM/c^2 for a finite result.")
    return math.sqrt(speed2)


# ---------------------------------------------------------------------------
# Exact orbit prediction from the constants of motion
# ---------------------------------------------------------------------------

class OrbitPrediction(NamedTuple):
    """What the constants of motion say about an orbit, before integrating it.

    kind is one of "bound", "circular (stable)", "circular (unstable)",
    "marginal", "plunge", "escape" or "radial".  A marginal orbit lies on
    the boundary between bound and plunging orbits and whirls ever closer to
    the unstable circular orbit at periapsis_radius.  Radii are areal radii in metres;
    advances are radians per radial period.  Values that do not apply to the
    kind are None.  A NamedTuple is immutable and, unlike a dataclass, stays
    importable when this module is loaded on its own from a file path.
    """
    kind: str
    periapsis_radius: Optional[float]
    apoapsis_radius: Optional[float]
    apsidal_advance: Optional[float]
    weak_field_advance: Optional[float]


def _complete_elliptic_k(m: float) -> float:
    """Complete elliptic integral K(m), m = k^2 in [0, 1), by the AGM."""
    if not 0.0 <= m < 1.0:
        raise ValueError("elliptic parameter must satisfy 0 <= m < 1.")
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(64):
        if abs(a - b) <= 4.0e-16 * a:
            break
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def _elliptic_k_excess(m: float) -> float:
    """Return (2/pi) K(m) - 1 without cancellation when m is small."""
    if m < 0.1:
        # (2/pi) K(m) = sum_n [C(2n, n) / 4^n]^2 m^n
        total, coefficient, power = 0.0, 1.0, 1.0
        for n in range(1, 60):
            coefficient *= (2 * n - 1) / (2 * n)
            power *= m
            term = coefficient * coefficient * power
            total += term
            if term <= 1.0e-18 * total:
                break
        return total
    return 2.0 * _complete_elliptic_k(m) / math.pi - 1.0


def weak_field_advance(periapsis_radius: float, apoapsis_radius: float) -> float:
    """Weak-field apsidal advance 6 pi GM / (c^2 p), p = semi-latus rectum."""
    _require_finite_real("periapsis_radius", periapsis_radius)
    _require_finite_real("apoapsis_radius", apoapsis_radius)
    semi_latus_rectum = 2.0 * periapsis_radius * apoapsis_radius / (
        periapsis_radius + apoapsis_radius
    )
    return 6.0 * math.pi * GM_SUN / (C2 * semi_latus_rectum)


def predict_orbit(
    x_init: float,
    u_init: float,
    model: str = "schwarzschild",
) -> OrbitPrediction:
    """Classify the orbit and give its turning points and exact apsidal advance.

    The particle starts at a turning point (its initial velocity is purely
    transverse).  With u = 1/r and h = x_init * u_init, the energy integral
    of the orbit equation gives

        (du/dphi)^2 = (2GM/c^2) u^3 - u^2 + (2GM/h^2) u + 2E/h^2

    (the cubic term only in Schwarzschild mode).  In the scaled variable
    w = u * x_init, so that the start is w = 1, this is

        (dw/dphi)^2 = A w^3 - w^2 + B w + (1 - A - B)
                    = (w - 1)(A w^2 + (A - 1) w + (A + B - 1)),

        A = 2GM/(c^2 x_init),   B = 2GM/(x_init u_init^2).

    The other roots decide whether the orbit is bound, plunges or escapes.
    For a bound Schwarzschild orbit between roots w1 < w2 with third root
    w3 = 1/A - w1 - w2, the azimuth swept per radial period is

        4 K(m) / sqrt(A (w3 - w1)),   m = (w2 - w1)/(w3 - w1),

    and the apsidal advance is that minus 2 pi.  This is exact for the
    equation the program integrates, so it is an outside check on the
    step-by-step orbit.
    """
    _require_finite_real("x_init", x_init)
    _require_finite_real("u_init", u_init)
    if x_init <= 0.0:
        raise ValueError("x_init must be positive.")
    key = _model_key(model)
    if key == "schwarzschild" and x_init <= HORIZON_RADIUS:
        raise ValueError("x_init must lie outside the Schwarzschild horizon.")
    if u_init == 0.0:
        return OrbitPrediction("radial", None, None, None, None)

    a = 2.0 * GM_SUN / (C2 * x_init) if key == "schwarzschild" else 0.0
    # So far from the mass that the cubic term cannot be represented: the
    # Newtonian turning points apply and the advance is the weak-field one.
    far = key == "schwarzschild" and a < 1.0e-200
    b = (2.0 * GM_SUN / x_init) / abs(u_init) / abs(u_init)
    if not math.isfinite(b):
        # The angular momentum is too small to matter: the particle falls
        # almost straight in.
        if key == "schwarzschild":
            return OrbitPrediction("plunge", None, x_init, None, None)
        return OrbitPrediction("bound", 0.0, x_init, 0.0, 0.0)

    slope = 3.0 * a - 2.0 + b  # d/dw of the right-hand side at w = 1
    q1 = a - 1.0
    q0 = a + b - 1.0
    if key == "schwarzschild" and not far:
        others = []
        disc = q1 * q1 - 4.0 * a * q0
        if disc >= 0.0:
            root = math.sqrt(disc)
            # Numerically stable pair of quadratic roots.
            qq = -0.5 * (q1 + math.copysign(root, q1))
            others = sorted((qq / a, q0 / qq if qq != 0.0 else math.inf))
    else:
        others = [-q0 / q1]  # = b - 1

    if any(abs(r - 1.0) <= 1.0e-9 for r in others):
        # w = 1 is a double root: a circular orbit.
        stable = key == "newtonian" or x_init > ISCO_RADIUS
        kind = "circular (stable)" if stable else "circular (unstable)"
        advance = weak = None
        if key == "newtonian":
            advance = weak = 0.0
        elif stable:
            # Small radial oscillations: 2 pi / sqrt(1 - 6GM/(c^2 r)) per period.
            advance = 2.0 * math.pi / math.sqrt(1.0 - 3.0 * a) - 2.0 * math.pi
            # The weak-field formula describes an oscillating orbit; an
            # unstable circular orbit has none, so it is given only here.
            weak = weak_field_advance(x_init, x_init)
        return OrbitPrediction(kind, x_init, x_init, advance, weak)

    if slope > 0.0:
        # w increases: the particle moves inward from an apoapsis.
        inner = [r for r in others if r > 1.0]
        if not inner:
            return OrbitPrediction("plunge", None, x_init, None, None)
        w1, w2 = 1.0, inner[0]
    else:
        # w decreases: the particle moves outward from a periapsis.
        outer = [r for r in others if r < 1.0]
        if not outer or outer[-1] <= 0.0:
            return OrbitPrediction("escape", x_init, None, None, None)
        w1, w2 = outer[-1], 1.0

    periapsis, apoapsis = x_init / w2, x_init / w1
    if key == "newtonian":
        return OrbitPrediction("bound", periapsis, apoapsis, 0.0, 0.0)
    weak = weak_field_advance(periapsis, apoapsis)
    if far:
        return OrbitPrediction("bound", periapsis, apoapsis, weak, weak)
    # sweep/(2 pi) = [(2/pi) K(m)] / sqrt(A (w3 - w1)), with w3 = 1/A - w1 - w2
    # because the three roots sum to 1/A, so A (w3 - w1) = 1 - A (2 w1 + w2).
    # Both factors are written as 1 + (small excess) so that a weak-field
    # advance does not vanish in the subtraction of 2 pi.
    w3 = 1.0 / a - w1 - w2
    if w3 - w2 <= 1.0e-9 * w2:
        # w2 is (to rounding) a double root: the orbit is on the boundary
        # between bound and plunging and whirls ever closer to the unstable
        # circular orbit at x_init / w2 without returning.
        return OrbitPrediction("marginal", periapsis, apoapsis, None, None)
    m = (w2 - w1) / (w3 - w1)
    k_excess = _elliptic_k_excess(m)
    root_excess = math.expm1(-0.5 * math.log1p(-a * (2.0 * w1 + w2)))
    advance = 2.0 * math.pi * (k_excess + root_excess + k_excess * root_excess)
    return OrbitPrediction("bound", periapsis, apoapsis, advance, weak)
