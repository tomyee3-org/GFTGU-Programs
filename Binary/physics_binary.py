"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import frexp, hypot, isfinite, ldexp
from numbers import Real

MODEL_VERSION = "1.1.2"


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
    or underflowing when the final result is representable as a float.
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

    # Normalize first and divide sequentially by r. This is algebraically
    # equivalent to displacement/r^3, but avoids constructing r^3 and thereby
    # supports the full useful floating-point separation range.
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
    K = KA + KB
    E = K + U
    values = (U, K, E)
    if not all(isfinite(value) for value in values):
        raise ValueError(
            "The calculated energy is outside the numerical range of "
            "double-precision arithmetic."
        )
    return values
