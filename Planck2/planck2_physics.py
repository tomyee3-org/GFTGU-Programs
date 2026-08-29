"""
planck2_physics.py

Physics for Planck2, accompanying the black-body radiation investigations
in Chapter 10 of Gravity from the Ground Up.

Planck2 works with three physical spectral quantities:

    "wavelength"      spectral radiance B_lambda per unit wavelength
    "frequency"       spectral radiance B_nu per unit frequency
    "energy_density"  spectral energy density u_nu per unit frequency

The calculation uses the dimensionless variable

    x = h c / (lambda k T) = h nu / (k T)

and dimensionless shape functions x^p/(exp(x)-1), with p=5 for B_lambda
and p=3 for the two frequency-based quantities.
"""

from dataclasses import dataclass
from typing import Literal
import math

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.0.3"
BUILD_ID_COVERS = (
    "planck2_physics.py",
    "planck2_driver.py",
    "main.py",
    "planck2_plot.py",
)


def _build_id_from_texts(source_texts) -> str:
    """Hash a mapping containing the normalized text of every core module.

    Line-ending normalization belongs to the file-reading boundary in
    ``_compute_build_id``.  Other textual differences, including a UTF-8 BOM
    or different Unicode normalization, intentionally identify a new build.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in BUILD_ID_COVERS:
        content = source_texts[name].encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def _compute_build_id(module_dir=None) -> str:
    """Return a short, reproducible identifier for the core source files.

    Files are read as UTF-8 text with universal-newline conversion, so merely
    switching between LF and CRLF line endings does not create a new build.
    Filename and byte-length framing prevents ambiguous concatenations.  A
    UTF-8 BOM and Unicode normalization changes are treated as content changes.
    """
    import os

    try:
        here = (
            os.path.dirname(os.path.abspath(__file__))
            if module_dir is None
            else os.path.abspath(os.fspath(module_dir))
        )
        source_texts = {}
        for name in BUILD_ID_COVERS:
            path = os.path.join(here, name)
            with open(path, "r", encoding="utf-8", newline=None) as source:
                source_texts[name] = source.read()
        return _build_id_from_texts(source_texts)
    except (KeyError, OSError, UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise RuntimeError(
            "Cannot compute BUILD_ID because a core source file is missing, "
            "unreadable, or not valid UTF-8."
        ) from exc


BUILD_ID = _compute_build_id()

# Physical constants (SI)
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZMANN = 1.380649e-23
WIEN_SCALE = H_PLANCK * C_LIGHT / K_BOLTZMANN
FREQUENCY_SCALE = K_BOLTZMANN / H_PLANCK
WAVELENGTH_PREFACTOR_SCALE = (
    2.0 * K_BOLTZMANN**5 / (H_PLANCK**4 * C_LIGHT**3)
)
FREQUENCY_PREFACTOR_SCALE = (
    2.0 * K_BOLTZMANN**3 / (H_PLANCK**2 * C_LIGHT**2)
)
ENERGY_DENSITY_PREFACTOR_SCALE = (
    8.0 * math.pi * K_BOLTZMANN**3 / (H_PLANCK**2 * C_LIGHT**3)
)
SIGMA_SB = (
    2.0 * math.pi**5 * K_BOLTZMANN**4
    / (15.0 * H_PLANCK**3 * C_LIGHT**2)
)

PlanckQuantity = Literal["wavelength", "frequency", "energy_density"]

SHAPE_EXPONENT = {
    "wavelength": 5,
    "frequency": 3,
    "energy_density": 3,
}


@dataclass(frozen=True)
class PlanckDomain:
    """Dimensionless x-domain and approximation boundaries."""
    x_min: float = 0.01
    x_max: float = 100.0
    x_low: float = 0.05
    x_high: float = 20.0

    def validate(self) -> None:
        values = (self.x_min, self.x_max, self.x_low, self.x_high)
        if not all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            for v in values
        ):
            raise ValueError("Domain values must be finite numbers.")
        if not (0.0 < self.x_min < self.x_max):
            raise ValueError("Require 0 < x_min < x_max.")
        if not (self.x_min <= self.x_low < self.x_high <= self.x_max):
            raise ValueError(
                "Require x_min <= x_low < x_high <= x_max."
            )


def validate_quantity(quantity: str) -> None:
    if not isinstance(quantity, str) or quantity not in SHAPE_EXPONENT:
        raise ValueError(
            'quantity must be "wavelength", "frequency", or "energy_density".'
        )


def _validate_temperature(T: float) -> None:
    if (
        not isinstance(T, (int, float))
        or isinstance(T, bool)
        or not math.isfinite(T)
        or T <= 0.0
    ):
        raise ValueError("Temperature must be a finite positive number.")


def _validate_x(x: float) -> None:
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not math.isfinite(x)
        or x <= 0.0
    ):
        raise ValueError("x must be a finite positive number.")


def _require_positive_finite_result(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{label} is outside the representable floating-point range "
            "for the supplied values."
        )
    return value


def _scaled_product(factors, label: str) -> float:
    """Evaluate a positive product of (value, integer power) pairs safely.

    ``frexp`` separates every factor into a bounded mantissa and a binary
    exponent. Combining those parts prevents an intermediate ``T**n`` or
    multiplication from overflowing when the complete result is representable.
    """
    mantissa = 1.0
    exponent = 0
    for value, power in factors:
        factor_mantissa, factor_exponent = math.frexp(value)
        mantissa *= factor_mantissa**power
        exponent += factor_exponent * power
        mantissa, adjustment = math.frexp(mantissa)
        exponent += adjustment

    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ValueError(
            f"{label} is outside the representable floating-point range "
            "for the supplied values."
        ) from exc
    return _require_positive_finite_result(result, label)


def _ln_shape_function_unchecked(x: float, p: int, domain: PlanckDomain) -> float:
    """Internal branch evaluator after x, p, and domain have been validated."""
    if x > domain.x_high:
        # exp(x) >> 1, so ln(exp(x)-1) ~= x.
        ln_denom = x
    elif x < domain.x_low:
        # exp(x)-1 ~= x, so ln(exp(x)-1) ~= ln(x).
        ln_denom = math.log(x)
    else:
        # expm1 retains precision for small positive x.
        ln_denom = math.log(math.expm1(x))

    return p * math.log(x) - ln_denom


def ln_shape_function(x: float, p: int, domain: PlanckDomain) -> float:
    """Return ln[x^p/(exp(x)-1)] for p=3 or p=5 on a valid domain."""
    _validate_x(x)
    if not isinstance(p, int) or isinstance(p, bool) or p not in (3, 5):
        raise ValueError("p must be the integer 3 or 5.")
    if not isinstance(domain, PlanckDomain):
        raise ValueError("domain must be a PlanckDomain instance.")
    domain.validate()
    return _ln_shape_function_unchecked(x, p, domain)


def shape_function(x: float, quantity: PlanckQuantity, domain: PlanckDomain) -> float:
    """Return the dimensionless shape function for the selected quantity."""
    validate_quantity(quantity)
    return math.exp(ln_shape_function(x, SHAPE_EXPONENT[quantity], domain))


def prefactor(quantity: PlanckQuantity, T: float) -> float:
    """Return the SI prefactor multiplying the selected dimensionless shape."""
    validate_quantity(quantity)
    _validate_temperature(T)

    scale = {
        "wavelength": WAVELENGTH_PREFACTOR_SCALE,
        "frequency": FREQUENCY_PREFACTOR_SCALE,
        "energy_density": ENERGY_DENSITY_PREFACTOR_SCALE,
    }[quantity]
    power = 5 if quantity == "wavelength" else 3
    return _scaled_product(
        ((scale, 1), (T, power)),
        "The spectral prefactor",
    )


def coordinate_jacobian(quantity: PlanckQuantity, x: float, T: float) -> float:
    """Return |d(lambda)/dx| or d(nu)/dx for integrating the physical spectrum."""
    validate_quantity(quantity)
    _validate_temperature(T)
    _validate_x(x)

    if quantity == "wavelength":
        # |d(lambda)/dx| = (hc/k) x^-2 T^-1
        factors = ((WIEN_SCALE, 1), (x, -2), (T, -1))
    else:
        # d(nu)/dx = (k/h) T
        factors = ((FREQUENCY_SCALE, 1), (T, 1))
    return _scaled_product(factors, "The coordinate Jacobian")


def exact_physical_integral(quantity: PlanckQuantity, T: float) -> float:
    """Exact bolometric integral for the selected physical spectral quantity."""
    validate_quantity(quantity)
    _validate_temperature(T)

    scale = (
        SIGMA_SB / math.pi
        if quantity in ("wavelength", "frequency")
        else 4.0 * SIGMA_SB / C_LIGHT
    )
    return _scaled_product(
        ((scale, 1), (T, 4)),
        "The bolometric integral",
    )


def x_to_wavelength(x: float, T: float) -> float:
    """lambda = hc/(x k T), in metres."""
    _validate_x(x)
    _validate_temperature(T)
    return _scaled_product(
        ((WIEN_SCALE, 1), (x, -1), (T, -1)),
        "Wavelength",
    )


def x_to_frequency(x: float, T: float) -> float:
    """nu = x k T/h, in hertz."""
    _validate_x(x)
    _validate_temperature(T)
    return _scaled_product(
        ((FREQUENCY_SCALE, 1), (x, 1), (T, 1)),
        "Frequency",
    )


def units_label(quantity: PlanckQuantity) -> tuple[str, str]:
    """Return x-axis and y-axis labels for the selected quantity."""
    validate_quantity(quantity)
    if quantity == "wavelength":
        return ("Wavelength (m)", r"Spectral radiance $B_\lambda$ (W m$^{-3}$ sr$^{-1}$)")
    if quantity == "frequency":
        return ("Frequency (Hz)", r"Spectral radiance $B_\nu$ (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
    return ("Frequency (Hz)", r"Spectral energy density $u_\nu$ (J m$^{-3}$ Hz$^{-1}$)")


def physical_integral_units(quantity: PlanckQuantity) -> str:
    """Units of the spectrum integrated over its physical horizontal coordinate."""
    validate_quantity(quantity)
    if quantity in ("wavelength", "frequency"):
        return "W m^-2 sr^-1"
    return "J m^-3"
