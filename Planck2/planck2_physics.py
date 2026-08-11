"""
planck2_physics.py

Physics for Planck2, built to accompany Investigation 10.2 ("Exploring
the Planck Function") and Investigation 10.3 ("Computing the Planck
function") of Gravity From the Ground Up.

--------------------------------------------------------------------
Physical background (Investigation 10.2)
--------------------------------------------------------------------
Equation 10.5 defines the dimensionless variable

    x = hc / (lambda * k * T)

and Equation 10.7 gives the spectral radiance per unit wavelength as

    F_lambda = (2 pi k^5 T^5 / (h^4 c^3)) * f(x),   f(x) = x^5/(e^x - 1)

so that all of the physics beyond a T^5 prefactor and physical
constants is carried by the dimensionless function f(x) (Eq. 10.8).

This module generalizes that idea to the three forms of Planck's law
requested for Planck2:

    * "wavelength"  -- spectral radiance per unit wavelength, B_lambda,
                        i.e. Eq. 10.7/10.10 above, proportional to
                        f(x) = x^5/(e^x - 1).
    * "frequency"   -- spectral radiance per unit frequency, B_nu,
                        which (via nu = c/lambda) is proportional
                        instead to x^3/(e^x - 1).
    * "energy_density" -- spectral energy density per unit frequency,
                        u_nu, which differs from B_nu only by a
                        factor of 4/c and so has the same x-dependence,
                        x^3/(e^x - 1), but a different physical
                        prefactor and units.

In all three cases we factor out the T-dependent prefactor and work
with the dimensionless shape function of x, exactly as Investigation
10.2 does; only the prefactor (and hence the physical y-axis) changes
between the three quantities.

--------------------------------------------------------------------
Numerical background (Investigation 10.3)
--------------------------------------------------------------------
Investigation 10.3 explains why the shape functions are evaluated via
their natural log rather than directly: e^x overflows for large x
(the text bounds this around x > ~200), so the program instead works
with ln(f(x)) = p*ln(x) - ln(e^x - 1) (Eq. 10.12, generalized here to
an exponent p that is 5 for the wavelength form and 3 for the
frequency/energy-density forms) and only exponentiates the result at
the end. Three regimes are used for ln(e^x - 1) itself:

    * x large (x > x_high): e^x >> 1, so ln(e^x - 1) ~= x directly
      (the second term of Eq. 10.12 is dropped).
    * x small (x < x_low): e^x - 1 ~= x (Eq. 10.13), so
      ln(e^x - 1) ~= ln(x).
    * x in between: no approximation is needed; ln(e^x - 1) is
      evaluated directly.

x_low and x_high are exposed as parameters (with the same defaults
Schutz uses, 0.01 and 100) rather than hardcoded thresholds, so a
user can see directly how the approximation boundaries affect the
result.
"""

from dataclasses import dataclass
from typing import Literal
import math

# ---------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------
H_PLANCK = 6.62607015e-34    # Planck constant, J s
C_LIGHT = 2.99792458e8       # speed of light, m/s
K_BOLTZMANN = 1.380649e-23   # Boltzmann constant, J/K

PlanckQuantity = Literal["wavelength", "frequency", "energy_density"]

# The exponent p in the shape function f(x) = x^p / (e^x - 1) for each
# quantity (Investigation 10.2's f(x) = x^5/(e^x-1) is the
# "wavelength" case; changing variables from lambda to nu turns the
# x^5 into x^3 for the frequency-based quantities).
SHAPE_EXPONENT = {
    "wavelength": 5,
    "frequency": 3,
    "energy_density": 3,
}


@dataclass
class PlanckDomain:
    """Domain and approximation-boundary parameters, in the dimensionless
    variable x = hc/(lambda k T), following Investigation 10.3."""
    x_min: float = 0.01
    x_max: float = 100.0
    x_low: float = 0.01
    x_high: float = 100.0


def ln_shape_function(x: float, p: int, domain: PlanckDomain) -> float:
    """
    Compute ln(f(x)) = p*ln(x) - ln(e^x - 1) for the shape function
    f(x) = x^p / (e^x - 1), using the three-regime approximation of
    Investigation 10.3 (Eq. 10.12 and 10.13) to avoid overflow in e^x
    for large x and loss of precision for small x.
    """
    if x > domain.x_high:
        # e^x >> 1: ln(e^x - 1) ~= x
        ln_denom = x
    elif x < domain.x_low:
        # e^x - 1 ~= x for |x| << 1: ln(e^x - 1) ~= ln(x)
        ln_denom = math.log(x)
    else:
        ln_denom = math.log(math.exp(x) - 1.0)

    return p * math.log(x) - ln_denom


def shape_function(x: float, quantity: PlanckQuantity, domain: PlanckDomain) -> float:
    """
    Dimensionless shape function f(x) for the requested quantity,
    computed via its log (Investigation 10.3) and then exponentiated.
    """
    p = SHAPE_EXPONENT[quantity]
    return math.exp(ln_shape_function(x, p, domain))


def prefactor(quantity: PlanckQuantity, T: float) -> float:
    """
    The physical, T-dependent prefactor multiplying the dimensionless
    shape function f(x), for each of the three quantities. Derived by
    substituting x = hc/(lambda k T) (Eq. 10.5/10.6) into the standard
    Planck-law expressions.

    "wavelength": spectral radiance per unit wavelength, B_lambda
        B_lambda(x) = (2 pi h c^2 / lambda^5) / (e^x - 1)
                    = (2 pi k^5 T^5 / (h^4 c^3)) * f(x)      (Eq. 10.7)
        units: W / (m^2 sr m)   [per unit wavelength]

    "frequency": spectral radiance per unit frequency, B_nu
        B_nu(x) = (2 h nu^3 / c^2) / (e^x - 1)
                = (2 k^3 T^3 / (h^2 c^2)) * f(x)
        units: W / (m^2 sr Hz)  [per unit frequency]

    "energy_density": spectral energy density per unit frequency, u_nu
        u_nu = (4 pi / c) * B_nu
             = (8 pi k^3 T^3 / (h^2 c^3)) * f(x)
        units: J / (m^3 Hz)     [per unit frequency]
    """
    if quantity == "wavelength":
        return 2.0 * math.pi * K_BOLTZMANN**5 * T**5 / (H_PLANCK**4 * C_LIGHT**3)
    elif quantity == "frequency":
        return 2.0 * K_BOLTZMANN**3 * T**3 / (H_PLANCK**2 * C_LIGHT**2)
    elif quantity == "energy_density":
        return 8.0 * math.pi * K_BOLTZMANN**3 * T**3 / (H_PLANCK**2 * C_LIGHT**3)
    else:
        raise ValueError(f"Unknown quantity: {quantity!r}")


def x_to_wavelength(x: float, T: float) -> float:
    """lambda = hc/(x k T), Eq. 10.6, in meters."""
    return H_PLANCK * C_LIGHT / (x * K_BOLTZMANN * T)


def x_to_frequency(x: float, T: float) -> float:
    """nu = x k T / h (from x = h nu / kT), in Hz."""
    return x * K_BOLTZMANN * T / H_PLANCK


def units_label(quantity: PlanckQuantity) -> tuple[str, str]:
    """Return (x-axis label, y-axis label) for the given quantity."""
    if quantity == "wavelength":
        return ("Wavelength (m)", r"Spectral radiance $B_\lambda$ (W m$^{-3}$ sr$^{-1}$)")
    elif quantity == "frequency":
        return ("Frequency (Hz)", r"Spectral radiance $B_\nu$ (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
    elif quantity == "energy_density":
        return ("Frequency (Hz)", r"Spectral energy density $u_\nu$ (J m$^{-3}$ Hz$^{-1}$)")
    else:
        raise ValueError(f"Unknown quantity: {quantity!r}")
