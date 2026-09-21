"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import frexp, fsum, hypot, isfinite, ldexp, pi, sqrt
from numbers import Real

MODEL_VERSION = "1.2.1"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_binary.py",
    "driver_binary.py",
    "main.py",
    "plot_binary.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same declared
    version.  The hash is independent of LF versus CRLF line endings and
    frames each file with its name and length so file-boundary changes cannot
    collide with an unchanged concatenated byte stream.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located or decoded, as can happen in some
    frozen or zipped distributions.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


G: float = 6.67430e-11  # m^3 kg^-1 s^-2


@dataclass
class BinaryState:
    t: float
    xA: float
    yA: float
    vA: float
    uA: float
    xB: float
    yB: float
    vB: float
    uB: float


@dataclass(frozen=True)
class OrbitalElements:
    """Initial Keplerian elements in the centre-of-mass frame (SI units)."""

    kind: str
    eccentricity: float
    relative_semimajor: float | None
    period: float | None
    relative_periapsis: float | None
    relative_apoapsis: float | None
    relative_speed_periapsis: float | None
    relative_speed_apoapsis: float | None


def orbital_elements(MA: float, MB: float, xA: float, yA: float,
                     vA: float, uA: float, xB: float, yB: float,
                     vB: float, uB: float) -> OrbitalElements:
    """Compute two-body conic elements from an initial relative state.

    The returned lengths and speeds concern the relative orbit. Multiply by
    the other body's mass fraction for either centre-of-mass orbit. Undefined
    apsides and periods on open trajectories are returned as None.
    """
    _positive_mass("MA", MA)
    _positive_mass("MB", MB)
    for name, value in (("vA", vA), ("uA", uA), ("vB", vB), ("uB", uB)):
        _finite_real(name, value)
    rx, ry, r = relative_displacement(xA, yA, xB, yB)
    vx, vy = vA - vB, uA - uB
    mu = G * (MA + MB)
    if not all(isfinite(z) for z in (vx, vy, mu)) or mu <= 0:
        raise ValueError("Orbital elements are outside the numerical range.")
    v2 = vx * vx + vy * vy
    rv = rx * vx + ry * vy
    h = rx * vy - ry * vx
    specific_energy = 0.5 * v2 - mu / r
    if not all(isfinite(z) for z in (v2, rv, h, specific_energy)):
        raise ValueError("Orbital elements are outside the numerical range.")
    ex = ((v2 - mu / r) * rx - rv * vx) / mu
    ey = ((v2 - mu / r) * ry - rv * vy) / mu
    eccentricity = hypot(ex, ey)
    if not isfinite(eccentricity):
        raise ValueError("Orbital elements are outside the numerical range.")
    if h == 0:
        # Zero angular momentum: a straight-line (radial) path, inward or outward,
        # has no periapsis or apsidal speed in the usual sense.
        return OrbitalElements("radial", eccentricity, None, None,
                               None, None, None, None)
    # Treat roundoff at escape energy as a parabola.
    energy_tolerance = 1e-12 * mu / r
    if abs(specific_energy) <= energy_tolerance:
        kind = "parabolic"
        eccentricity = 1.0
        semi = period = apo = speed_apo = None
    elif specific_energy < 0:
        kind = "elliptic"
        semi = -mu / (2 * specific_energy)
        eccentricity = min(eccentricity, 1.0)
        period = 2 * pi * sqrt(semi ** 3 / mu)
        apo = semi * (1 + eccentricity)
        speed_apo = abs(h) / apo
    else:
        kind = "hyperbolic"
        semi = -mu / (2 * specific_energy)  # Signed conic semimajor axis.
        period = apo = speed_apo = None
    peri = h * h / (mu * (1 + eccentricity))
    speed_peri = abs(h) / peri
    if not all(isfinite(z) for z in (eccentricity, peri, speed_peri)):
        raise ValueError("Orbital elements are outside the numerical range.")
    return OrbitalElements(kind, eccentricity, semi, period,
                           peri, apo, speed_peri, speed_apo)


def _finite_real(name: str, value: float) -> None:
    """Reject values that cannot represent a finite physical input."""
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def _positive_mass(name: str, value: float) -> None:
    """Validate a point mass used by a public physics calculation."""
    _finite_real(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _scaled_positive_product_quotient(factors, divisors, quantity: str) -> float:
    """Evaluate a positive product and quotient without false range loss.

    All factors and divisors are represented as mantissa/exponent pairs before
    they are combined. This prevents an intermediate operation from overflowing
    or underflowing when the final result is representable as a float. Callers
    must supply only finite, strictly positive factors and divisors; public
    functions validate those preconditions before reaching this private helper.
    """
    mantissa = 1.0
    exponent = 0
    for factor in factors:
        factor_mantissa, factor_exponent = frexp(factor)
        mantissa *= factor_mantissa
        exponent += factor_exponent

    for divisor in divisors:
        divisor_mantissa, divisor_exponent = frexp(divisor)
        mantissa /= divisor_mantissa
        exponent -= divisor_exponent
    mantissa, adjustment = frexp(mantissa)
    exponent += adjustment

    try:
        result = ldexp(mantissa, exponent)
    except OverflowError as error:
        raise ValueError(
            f"The calculated {quantity} is outside the numerical range of "
            "double-precision arithmetic."
        ) from error
    if result == 0.0:
        raise ValueError(
            f"The calculated {quantity} is outside the numerical range of "
            "double-precision arithmetic."
        )
    return result


def _scaled_kinetic_energy(mass: float, velocity_x: float,
                           velocity_y: float) -> float:
    """Return one body's kinetic energy without squaring a huge velocity."""
    velocity_scale = max(abs(velocity_x), abs(velocity_y))
    if velocity_scale == 0.0:
        return 0.0

    scaled_x = velocity_x / velocity_scale
    scaled_y = velocity_y / velocity_scale
    scaled_speed_squared = scaled_x * scaled_x + scaled_y * scaled_y
    return _scaled_positive_product_quotient(
        (0.5, mass, velocity_scale, velocity_scale, scaled_speed_squared),
        (),
        "energy",
    )


def _finite_energy_sum(terms) -> float:
    """Accurately sum finite energy terms and reject a final overflow."""
    try:
        result = fsum(terms)
    except OverflowError as error:
        raise ValueError(
            "The calculated energy is outside the numerical range of "
            "double-precision arithmetic."
        ) from error
    if not isfinite(result):
        raise ValueError(
            "The calculated energy is outside the numerical range of "
            "double-precision arithmetic."
        )
    return result


def relative_displacement(xA: float, yA: float, xB: float, yB: float):
    """Return A-minus-B displacement components and scalar separation."""
    for name, value in (("xA", xA), ("yA", yA), ("xB", xB), ("yB", yB)):
        _finite_real(name, value)

    xAB = xA - xB
    yAB = yA - yB
    if not (isfinite(xAB) and isfinite(yAB)):
        raise ValueError(
            "The relative displacement is outside the numerical range of "
            "double-precision arithmetic."
        )

    rAB = hypot(xAB, yAB)

    if rAB == 0.0:
        raise ValueError(
            "The two point masses have zero separation. "
            "Newtonian point-mass gravity is singular at r = 0."
        )
    if not isfinite(rAB):
        raise ValueError(
            "The relative displacement is outside the numerical range of "
            "double-precision arithmetic."
        )

    return xAB, yAB, rAB


def accelerations(
    MA: float, MB: float,
    xA: float, yA: float,
    xB: float, yB: float,
):
    """Return the Newtonian acceleration components of bodies A and B."""
    _positive_mass("MA", MA)
    _positive_mass("MB", MB)
    xAB, yAB, rAB = relative_displacement(xA, yA, xB, yB)

    # Normalize the displacement for direction, then evaluate each inverse-square
    # magnitude with scaled mantissa/exponent arithmetic. This avoids both r^3
    # overflow and fixed-order intermediate overflow or underflow.
    direction_x = xAB / rAB
    direction_y = yAB / rAB
    acceleration_a = _scaled_positive_product_quotient(
        (G, MB), (rAB, rAB), "acceleration"
    )
    acceleration_b = _scaled_positive_product_quotient(
        (G, MA), (rAB, rAB), "acceleration"
    )
    axA = -acceleration_a * direction_x
    ayA = -acceleration_a * direction_y
    axB = acceleration_b * direction_x
    ayB = acceleration_b * direction_y
    values = (axA, ayA, axB, ayB)
    if not all(isfinite(value) for value in values):
        raise ValueError(
            "The calculated acceleration is outside the numerical range of "
            "double-precision arithmetic."
        )
    if (axA == 0.0 and ayA == 0.0) or (axB == 0.0 and ayB == 0.0):
        raise ValueError(
            "The calculated acceleration is outside the numerical range of "
            "double-precision arithmetic."
        )
    return values


def energies(
    MA: float, MB: float,
    xA: float, yA: float, vA: float, uA: float,
    xB: float, yB: float, vB: float, uB: float,
):
    """Return gravitational potential, kinetic, and total system energy."""
    _positive_mass("MA", MA)
    _positive_mass("MB", MB)
    for name, value in (("vA", vA), ("uA", uA), ("vB", vB), ("uB", uB)):
        _finite_real(name, value)
    _, _, rAB = relative_displacement(xA, yA, xB, yB)

    # Combine mantissas and binary exponents separately so neither overflow nor
    # underflow in an intermediate operation destroys a representable result.
    U = -_scaled_positive_product_quotient(
        (G, MA, MB), (rAB,), "energy"
    )
    KA = _scaled_kinetic_energy(MA, vA, uA)
    KB = _scaled_kinetic_energy(MB, vB, uB)
    K = _finite_energy_sum((KA, KB))
    # Sum all three independently computed terms together. Forming K + U could
    # discard a tiny KB while forming K, before a large KA cancels with U.
    E = _finite_energy_sum((KA, KB, U))
    values = (U, K, E)
    if not all(isfinite(value) for value in values):
        raise ValueError(
            "The calculated energy is outside the numerical range of "
            "double-precision arithmetic."
        )
    return values
