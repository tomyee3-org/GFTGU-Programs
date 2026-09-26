"""
Newtonian two-body physics for Binary.
"""

from dataclasses import dataclass
from math import frexp, fsum, hypot, isfinite, ldexp, pi, sqrt
from numbers import Real

MODEL_VERSION = "1.2.4"


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
    total_mass = MA + MB
    # Sum the two gravitational parameters separately when the mass sum itself
    # would overflow although G times it is representable.
    mu = G * total_mass if isfinite(total_mass) else G * MA + G * MB
    if not all(isfinite(z) for z in (vx, vy, mu)) or mu <= 0:
        raise ValueError("Orbital elements are outside the numerical range.")
    v2 = _scaled_sum_of_squares(vx, vy)
    h = _scaled_cross(rx, ry, vx, vy)
    if h == 0.0:
        # Zero angular momentum is a straight-line (radial) path. Classification
        # does not need the radial product r·v, which can overflow a double
        # even when every input is finite and the requested e = 1 is exact.
        return OrbitalElements("radial", 1.0, None, None,
                               None, None, None, None)
    rv = _scaled_dot(rx, ry, vx, vy)
    specific_energy = _specific_orbital_energy(v2, mu, r)
    if not all(isfinite(z) for z in (v2, rv, h, specific_energy)):
        raise ValueError("Orbital elements are outside the numerical range.")
    eccentricity = _eccentricity_magnitude(v2, mu, r, rx, ry, rv, vx, vy, h, specific_energy)
    # Treat roundoff at escape energy as a parabola.
    energy_tolerance = 1e-12 * mu / r
    if not isfinite(energy_tolerance):
        energy_tolerance = 0.0
    if abs(specific_energy) <= energy_tolerance:
        kind = "parabolic"
        eccentricity = 1.0
        semi = period = apo = speed_apo = None
    elif specific_energy < 0:
        kind = "elliptic"
        semi = _scaled_positive_product_quotient(
            (mu,), (2.0, abs(specific_energy)), "semi-major axis"
        )
        eccentricity = min(eccentricity, 1.0)
        # semi ** 3 could overflow when the period is representable.
        period = _elliptic_period(semi, mu)
        apo = _scaled_product(semi, 1.0 + eccentricity, "apoapsis")
        speed_apo = _safe_quotient(abs(h), apo, "apoapsis speed")
    else:
        kind = "hyperbolic"
        semi = -_scaled_positive_product_quotient(
            (mu,), (2.0, abs(specific_energy)), "semi-major axis"
        )
        period = apo = speed_apo = None
    peri = _periapsis_from_h(h, mu, eccentricity)
    # A denormal periapsis is a needle ellipse: |h|/peri is not a useful speed.
    if peri == 0.0 or abs(h) / peri == float("inf"):
        speed_peri = None
        if peri != 0.0 and peri < 1e-300:
            peri = 0.0
    else:
        try:
            speed_peri = _safe_quotient(abs(h), peri, "periapsis speed")
        except ValueError:
            speed_peri = None
    if not all(isfinite(z) for z in (eccentricity, peri, speed_peri, period, apo, speed_apo, semi)
               if z is not None):
        raise ValueError("Orbital elements are outside the numerical range.")
    return OrbitalElements(kind, eccentricity, semi, period,
                           peri, apo, speed_peri, speed_apo)


def _scaled_sum_of_squares(x: float, y: float) -> float:
    """Return x*x + y*y without a false intermediate overflow."""
    scale = max(abs(x), abs(y))
    if scale == 0.0:
        return 0.0
    xs, ys = x / scale, y / scale
    return (xs * xs + ys * ys) * scale * scale


def _scaled_dot(ax: float, ay: float, bx: float, by: float) -> float:
    """Return ax*bx + ay*by, or inf if the finite product overflows."""
    try:
        return _signed_scaled_product(ax, bx) + _signed_scaled_product(ay, by)
    except ValueError:
        return float("inf") if (ax * bx + ay * by) >= 0 else float("-inf")


def _scaled_cross(ax: float, ay: float, bx: float, by: float) -> float:
    """Return ax*by - ay*bx, or inf if the finite product overflows."""
    try:
        return _signed_scaled_product(ax, by) - _signed_scaled_product(ay, bx)
    except ValueError:
        left = ax * by
        right = ay * bx
        return float("inf") if (left - right) >= 0 else float("-inf")


def _signed_scaled_product(a: float, b: float) -> float:
    """Return a*b for finite a, b, raising ValueError only if the product overflows."""
    if a == 0.0 or b == 0.0:
        return 0.0
    sign = -1.0 if (a < 0.0) ^ (b < 0.0) else 1.0
    return sign * _scaled_positive_product_quotient((abs(a), abs(b)), (), "product")


def _specific_orbital_energy(v2: float, mu: float, r: float) -> float:
    """Return v^2/2 - mu/r without losing a representable result to mu/r overflow."""
    kinetic = 0.5 * v2
    try:
        potential = _scaled_positive_product_quotient((mu,), (r,), "specific potential")
    except ValueError:
        return float("-inf") if isfinite(kinetic) else float("nan")
    energy = kinetic - potential
    if isfinite(energy):
        return energy
    if isfinite(kinetic) and potential > abs(kinetic):
        return -potential
    return energy


def _eccentricity_magnitude(v2, mu, r, rx, ry, rv, vx, vy, h, specific_energy) -> float:
    """Return |e| from the eccentricity vector, or from 1 + 2 E h^2 / mu^2."""
    try:
        ex = _eccentricity_component(v2, mu, r, rx, rv, vx)
        ey = _eccentricity_component(v2, mu, r, ry, rv, vy)
        eccentricity = hypot(ex, ey)
        if isfinite(eccentricity):
            return eccentricity
    except ValueError:
        pass
    return _eccentricity_from_energy(specific_energy, h, mu)


def _eccentricity_component(v2, mu, r, position, rv, velocity) -> float:
    """One Cartesian component of the eccentricity vector, ((v^2-mu/r) r - (r·v) v)/mu."""
    first = _signed_scaled_product(v2 - mu / r if isfinite(mu / r) else v2, position)
    if not isfinite(mu / r) and position != 0.0:
        # (v^2 - mu/r) * position = v^2 * position - mu * position/r
        first = _signed_scaled_product(v2, position) - _signed_scaled_product(mu, position / r)
    second = _signed_scaled_product(rv, velocity)
    numerator = first - second
    if numerator == 0.0:
        return 0.0
    sign = -1.0 if numerator < 0.0 else 1.0
    return sign * _scaled_positive_product_quotient((abs(numerator),), (mu,), "eccentricity")


def _eccentricity_from_energy(specific_energy, h, mu) -> float:
    """e = sqrt(1 + 2 E h^2 / mu^2), evaluated so huge e still fits when it can."""
    if mu <= 0.0:
        raise ValueError("Orbital elements are outside the numerical range.")
    alpha = _scaled_positive_product_quotient(
        (sqrt(2.0), sqrt(abs(specific_energy)) if specific_energy != 0.0 else 0.0, abs(h)),
        (mu,),
        "eccentricity",
    ) if specific_energy != 0.0 else 0.0
    if specific_energy >= 0.0:
        return hypot(1.0, alpha)
    inner = 1.0 - alpha * alpha
    if inner <= 0.0:
        return 0.0
    return sqrt(inner)


def _elliptic_period(semi, mu) -> float:
    """2 pi sqrt(a^3 / mu), without cubing a when that overflows."""
    if not isfinite(semi) or semi <= 0.0:
        raise ValueError("Orbital elements are outside the numerical range.")
    root = sqrt(_scaled_positive_product_quotient((semi,), (mu,), "period"))
    return 2 * pi * _scaled_positive_product_quotient((semi, root), (), "period")


def _scaled_product(a, b, quantity) -> float:
    if a == 0.0 or b == 0.0:
        return 0.0
    sign = -1.0 if (a < 0.0) ^ (b < 0.0) else 1.0
    return sign * _scaled_positive_product_quotient((abs(a), abs(b)), (), quantity)


def _periapsis_from_h(h, mu, eccentricity) -> float:
    """h^2 / (mu (1+e)). Underflow to zero is kept; overflow is a range error."""
    one_plus_e = 1.0 + eccentricity
    if not isfinite(one_plus_e) or one_plus_e <= 0.0:
        raise ValueError("Orbital elements are outside the numerical range.")
    try:
        return _scaled_positive_product_quotient(
            (abs(h), abs(h)), (mu, one_plus_e), "periapsis"
        )
    except ValueError as error:
        if "outside the numerical range" in str(error):
            # A periapsis of zero is a needle ellipse, not a failed classification.
            if abs(h) < 1.0 and mu > 1.0:
                return 0.0
        raise


def _safe_quotient(numerator, denominator, quantity) -> float:
    if denominator == 0.0:
        raise ValueError("Orbital elements are outside the numerical range.")
    sign = -1.0 if numerator < 0.0 else 1.0
    return sign * _scaled_positive_product_quotient(
        (abs(numerator),), (abs(denominator),), quantity
    )


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
