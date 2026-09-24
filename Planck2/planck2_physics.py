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
from types import MappingProxyType
from typing import Literal
import math

# Public release metadata. MODEL_VERSION changes when the model's documented
# behaviour changes; BUILD_ID changes whenever one of the core source files
# changes.
MODEL_VERSION = "1.5.1"
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


@dataclass(frozen=True)
class QuantitySpec:
    """Everything that differs from one spectral quantity to the next.

    This is the single place where the shape exponent, prefactor scale,
    exact bolometric scale, horizontal coordinate, axis labels, and integral
    units of a quantity are recorded.  The driver, the plotter, and the
    module-level helper functions all read from it so the metadata cannot
    drift apart.
    """
    name: str
    shape_exponent: int
    coordinate: str
    coordinate_symbol: str
    coordinate_unit: str
    prefactor_scale: float
    exact_integral_scale: float
    x_label: str
    y_label: str
    integral_units: str


_SPEC_TABLE = {
    "wavelength": QuantitySpec(
        name="wavelength",
        shape_exponent=5,
        coordinate="wavelength",
        coordinate_symbol="\u03bb",
        coordinate_unit="m",
        prefactor_scale=WAVELENGTH_PREFACTOR_SCALE,
        exact_integral_scale=SIGMA_SB / math.pi,
        x_label="Wavelength (m)",
        y_label=r"Spectral radiance $B_\lambda$ (W m$^{-3}$ sr$^{-1}$)",
        integral_units="W m^-2 sr^-1",
    ),
    "frequency": QuantitySpec(
        name="frequency",
        shape_exponent=3,
        coordinate="frequency",
        coordinate_symbol="\u03bd",
        coordinate_unit="Hz",
        prefactor_scale=FREQUENCY_PREFACTOR_SCALE,
        exact_integral_scale=SIGMA_SB / math.pi,
        x_label="Frequency (Hz)",
        y_label=r"Spectral radiance $B_\nu$ (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)",
        integral_units="W m^-2 sr^-1",
    ),
    "energy_density": QuantitySpec(
        name="energy_density",
        shape_exponent=3,
        coordinate="frequency",
        coordinate_symbol="\u03bd",
        coordinate_unit="Hz",
        prefactor_scale=ENERGY_DENSITY_PREFACTOR_SCALE,
        exact_integral_scale=4.0 * SIGMA_SB / C_LIGHT,
        x_label="Frequency (Hz)",
        y_label=r"Spectral energy density $u_\nu$ (J m$^{-3}$ Hz$^{-1}$)",
        integral_units="J m^-3",
    ),
}

# Both mappings are read-only, so no caller can make one part of the program
# disagree with another about a quantity.
QUANTITY_SPECS = MappingProxyType(_SPEC_TABLE)

# Read-only snapshot for callers that only need the shape exponent.
SHAPE_EXPONENT = MappingProxyType(
    {name: spec.shape_exponent for name, spec in _SPEC_TABLE.items()}
)


def quantity_spec(quantity: str) -> QuantitySpec:
    """Return the :class:`QuantitySpec` for a valid quantity name."""
    validate_quantity(quantity)
    return QUANTITY_SPECS[quantity]


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
    if not isinstance(quantity, str) or quantity not in QUANTITY_SPECS:
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

    ``factors`` must be a nonempty iterable of pairs.  Every value must be a
    finite positive number and every power must be an integer (zero and
    negative powers are allowed).  ``label`` must be a nonempty string.

    ``frexp`` separates every factor into a bounded mantissa and a binary
    exponent. Combining those parts prevents an intermediate ``T**n`` or
    multiplication from overflowing when the complete result is representable.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a nonempty string.")

    try:
        factor_iterator = iter(factors)
    except TypeError as exc:
        raise ValueError("factors must be a nonempty iterable of pairs.") from exc

    def multiply_scaled(left, right):
        left_mantissa, left_exponent = left
        right_mantissa, right_exponent = right
        product_mantissa, adjustment = math.frexp(
            left_mantissa * right_mantissa
        )
        return (
            product_mantissa,
            left_exponent + right_exponent + adjustment,
        )

    def integer_power_scaled(value, power):
        base_mantissa, base_exponent = math.frexp(value)
        if power < 0:
            base_mantissa, adjustment = math.frexp(1.0 / base_mantissa)
            base_exponent = adjustment - base_exponent
            power = -power

        product = (0.5, 1)  # frexp representation of 1.0
        base = (base_mantissa, base_exponent)
        while power:
            if power & 1:
                product = multiply_scaled(product, base)
            power //= 2
            if power:
                base = multiply_scaled(base, base)
        return product

    scaled_product = (0.5, 1)
    factor_count = 0
    for pair in factor_iterator:
        try:
            value, power = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("Every factor must be a (value, power) pair.") from exc
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isinstance(power, int)
            or isinstance(power, bool)
        ):
            raise ValueError(
                "Factor values must be finite positive numbers and powers "
                "must be integers."
            )
        try:
            value_is_valid = math.isfinite(value) and value > 0.0
        except OverflowError:
            value_is_valid = False
        if not value_is_valid:
            raise ValueError(
                "Factor values must be finite positive numbers and powers "
                "must be integers."
            )

        factor_count += 1
        scaled_product = multiply_scaled(
            scaled_product,
            integer_power_scaled(value, power),
        )

    if factor_count == 0:
        raise ValueError("factors must be a nonempty iterable of pairs.")

    mantissa, exponent = scaled_product

    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ValueError(
            f"{label} is outside the representable floating-point range "
            "for the supplied values."
        ) from exc
    return _require_positive_finite_result(result, label)


# math.expm1 overflows a little above 709.78.
_EXPM1_SAFE_LIMIT = 700.0


def _ln_shape_function_unchecked(x: float, p: int, domain: PlanckDomain) -> float:
    """Internal branch evaluator after x, p, and domain have been validated."""
    if x > domain.x_high:
        # exp(x) >> 1, so ln(exp(x)-1) ~= x.
        ln_denom = x
    elif x < domain.x_low:
        # exp(x)-1 ~= x, so ln(exp(x)-1) ~= ln(x).
        ln_denom = math.log(x)
    else:
        # expm1 retains precision for small positive x.  Beyond the point
        # where expm1 itself overflows, use the identical algebraic form
        # x + ln(1 - exp(-x)), which stays finite for every representable x.
        if x > _EXPM1_SAFE_LIMIT:
            ln_denom = x + math.log1p(-math.exp(-x))
        else:
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
    return math.exp(
        ln_shape_function(x, QUANTITY_SPECS[quantity].shape_exponent, domain)
    )


def prefactor(quantity: PlanckQuantity, T: float) -> float:
    """Return the SI prefactor multiplying the selected dimensionless shape."""
    validate_quantity(quantity)
    _validate_temperature(T)

    spec = QUANTITY_SPECS[quantity]
    return _scaled_product(
        ((spec.prefactor_scale, 1), (T, spec.shape_exponent)),
        "The spectral prefactor",
    )


def coordinate_jacobian(quantity: PlanckQuantity, x: float, T: float) -> float:
    """Return |d(lambda)/dx| or d(nu)/dx for integrating the physical spectrum."""
    validate_quantity(quantity)
    _validate_temperature(T)
    _validate_x(x)

    if QUANTITY_SPECS[quantity].coordinate == "wavelength":
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

    return _scaled_product(
        ((QUANTITY_SPECS[quantity].exact_integral_scale, 1), (T, 4)),
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
    spec = quantity_spec(quantity)
    return (spec.x_label, spec.y_label)


def physical_integral_units(quantity: PlanckQuantity) -> str:
    """Units of the spectrum integrated over its physical horizontal coordinate."""
    return quantity_spec(quantity).integral_units
