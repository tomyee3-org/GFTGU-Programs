"""
planck2_driver.py

Driver for Planck2. Samples the requested Planck-law quantity
(wavelength radiance, frequency radiance, or energy density) across
the dimensionless domain in x, following the numerical method of
Investigation 10.3 (peak-finding by direct comparison, area by the
trapezoidal rule -- see Figure 10.5), then converts x back to physical
units (wavelength or frequency) for plotting.

n_steps directly and functionally controls the sampling resolution:
it sets the number of divisions of the dimensionless domain
[x_min, x_max], so a larger n_steps gives a finer sampling grid and a
more accurate peak location and area.
"""

from dataclasses import dataclass
from typing import List, Optional
import math

from planck2_physics import (
    PlanckDomain,
    PlanckQuantity,
    ln_shape_function,
    prefactor,
    x_to_wavelength,
    x_to_frequency,
    units_label,
    SHAPE_EXPONENT,
)


@dataclass
class Planck2Result:
    quantity: PlanckQuantity
    T: float
    x_values: List[float]
    coord_values: List[float]   # wavelength (m) or frequency (Hz), depending on quantity
    y_values: List[float]       # physical spectral radiance / energy density
    x_peak: float
    coord_peak: float
    y_peak: float
    area: float                 # area under the dimensionless shape curve, i.e. integral of f(x) dx
    x_label: str
    y_label: str


def run_planck2(
    T: float,
    quantity: PlanckQuantity,
    n_steps: int = 2000,
    domain: Optional[PlanckDomain] = None,
) -> Planck2Result:
    """
    Run the Planck2 computation.

    Parameters
    ----------
    T : float
        Temperature in kelvin.
    quantity : "wavelength", "frequency", or "energy_density"
        Which form of Planck's law to compute.
    n_steps : int
        Number of divisions of the dimensionless domain x in
        [domain.x_min, domain.x_max]. Directly controls the sampling
        resolution.
    domain : PlanckDomain, optional
        Domain and small-x/large-x approximation boundaries. Defaults
        to (x_min=0.01, x_max=100, x_low=0.01, x_high=100).

    Returns
    -------
    Planck2Result
    """
    if domain is None:
        domain = PlanckDomain()
    if T <= 0.0:
        raise ValueError("Temperature must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    dx = (domain.x_max - domain.x_min) / n_steps
    pref = prefactor(quantity, T)
    p = SHAPE_EXPONENT[quantity]

    x_values: List[float] = []
    coord_values: List[float] = []
    y_values: List[float] = []

    x_peak = 0.0
    y_peak = -1.0  # any real shape-function value is >= 0, so this is a safe seed
    coord_peak = 0.0

    area = 0.0
    y_last = 0.0

    x = domain.x_min
    for _ in range(n_steps + 1):
        ln_f = ln_shape_function(x, p, domain)
        f = math.exp(ln_f)
        y = pref * f

        if quantity in ("wavelength",):
            coord = x_to_wavelength(x, T)
        else:
            coord = x_to_frequency(x, T)

        x_values.append(x)
        coord_values.append(coord)
        y_values.append(y)

        if f > y_peak:
            y_peak = f
            x_peak = x
            coord_peak = coord

        area += 0.5 * (f + y_last) * dx
        y_last = f

        x += dx

    x_label, y_label = units_label(quantity)

    return Planck2Result(
        quantity=quantity,
        T=T,
        x_values=x_values,
        coord_values=coord_values,
        y_values=y_values,
        x_peak=x_peak,
        coord_peak=coord_peak,
        y_peak=pref * y_peak,
        area=area,
        x_label=x_label,
        y_label=y_label,
    )
