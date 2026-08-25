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
MODEL_VERSION = "1.0.0"
BUILD_ID_COVERS = (
    "planck2_physics.py",
    "planck2_driver.py",
    "main.py",
    "planck2_plot.py",
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

# Physical constants (SI)
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZMANN = 1.380649e-23
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
    if quantity not in SHAPE_EXPONENT:
        raise ValueError(
            'quantity must be "wavelength", "frequency", or "energy_density".'
        )


def ln_shape_function(x: float, p: int, domain: PlanckDomain) -> float:
    """Return ln[x^p/(exp(x)-1)] using stable small/exact/large-x branches."""
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0.0:
        raise ValueError("x must be a finite positive number.")

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


def shape_function(x: float, quantity: PlanckQuantity, domain: PlanckDomain) -> float:
    """Return the dimensionless shape function for the selected quantity."""
    validate_quantity(quantity)
    return math.exp(ln_shape_function(x, SHAPE_EXPONENT[quantity], domain))


def prefactor(quantity: PlanckQuantity, T: float) -> float:
    """Return the SI prefactor multiplying the selected dimensionless shape."""
    validate_quantity(quantity)
    if not isinstance(T, (int, float)) or isinstance(T, bool) or not math.isfinite(T) or T <= 0.0:
        raise ValueError("Temperature must be a finite positive number.")

    if quantity == "wavelength":
        # B_lambda = [2 k^5 T^5/(h^4 c^3)] x^5/(exp(x)-1)
        return 2.0 * K_BOLTZMANN**5 * T**5 / (H_PLANCK**4 * C_LIGHT**3)
    if quantity == "frequency":
        # B_nu = [2 k^3 T^3/(h^2 c^2)] x^3/(exp(x)-1)
        return 2.0 * K_BOLTZMANN**3 * T**3 / (H_PLANCK**2 * C_LIGHT**2)

    # u_nu = (4 pi/c) B_nu
    return 8.0 * math.pi * K_BOLTZMANN**3 * T**3 / (H_PLANCK**2 * C_LIGHT**3)


def coordinate_jacobian(quantity: PlanckQuantity, x: float, T: float) -> float:
    """Return |d(lambda)/dx| or d(nu)/dx for integrating the physical spectrum."""
    validate_quantity(quantity)
    if not isinstance(T, (int, float)) or isinstance(T, bool) or not math.isfinite(T) or T <= 0.0:
        raise ValueError("Temperature must be a finite positive number.")
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0.0:
        raise ValueError("x must be a finite positive number.")

    if quantity == "wavelength":
        # lambda = hc/(x k T)
        return H_PLANCK * C_LIGHT / (K_BOLTZMANN * T * x * x)
    # nu = x k T / h
    return K_BOLTZMANN * T / H_PLANCK


def exact_physical_integral(quantity: PlanckQuantity, T: float) -> float:
    """Exact bolometric integral for the selected physical spectral quantity."""
    validate_quantity(quantity)
    if not isinstance(T, (int, float)) or isinstance(T, bool) or not math.isfinite(T) or T <= 0.0:
        raise ValueError("Temperature must be a finite positive number.")

    if quantity in ("wavelength", "frequency"):
        # Integral of spectral radiance over wavelength or frequency.
        return SIGMA_SB * T**4 / math.pi
    # Total black-body energy density.
    return 4.0 * SIGMA_SB * T**4 / C_LIGHT


def x_to_wavelength(x: float, T: float) -> float:
    """lambda = hc/(x k T), in metres."""
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not math.isfinite(x)
        or x <= 0.0
        or not isinstance(T, (int, float))
        or isinstance(T, bool)
        or not math.isfinite(T)
        or T <= 0.0
    ):
        raise ValueError("x and T must be finite positive numbers.")
    return H_PLANCK * C_LIGHT / (x * K_BOLTZMANN * T)


def x_to_frequency(x: float, T: float) -> float:
    """nu = x k T/h, in hertz."""
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not math.isfinite(x)
        or x <= 0.0
        or not isinstance(T, (int, float))
        or isinstance(T, bool)
        or not math.isfinite(T)
        or T <= 0.0
    ):
        raise ValueError("x and T must be finite positive numbers.")
    return x * K_BOLTZMANN * T / H_PLANCK


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
