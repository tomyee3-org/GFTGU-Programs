"""Driver for Planck2."""

from dataclasses import dataclass
from typing import List, Optional
import math

from planck2_physics import (
    PlanckDomain,
    PlanckQuantity,
    SHAPE_EXPONENT,
    coordinate_jacobian,
    exact_physical_integral,
    ln_shape_function,
    physical_integral_units,
    prefactor,
    units_label,
    validate_quantity,
    x_to_frequency,
    x_to_wavelength,
)


@dataclass
class Planck2Result:
    quantity: PlanckQuantity
    T: float
    x_values: List[float]
    coord_values: List[float]
    y_values: List[float]
    x_peak: float
    coord_peak: float
    y_peak: float
    dimensionless_area: float
    physical_integral: float
    exact_physical_integral: float
    physical_integral_units: str
    x_label: str
    y_label: str

    @property
    def area(self) -> float:
        """Backward-compatible alias for the dimensionless shape-function area."""
        return self.dimensionless_area


def run_planck2(
    T: float,
    quantity: PlanckQuantity,
    n_steps: int = 2000,
    domain: Optional[PlanckDomain] = None,
) -> Planck2Result:
    """Sample the selected Planck spectrum on a uniform grid in dimensionless x."""
    if domain is None:
        domain = PlanckDomain()

    validate_quantity(quantity)
    domain.validate()
    if T <= 0.0:
        raise ValueError("Temperature must be positive.")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")

    dx = (domain.x_max - domain.x_min) / n_steps
    pref = prefactor(quantity, T)
    p = SHAPE_EXPONENT[quantity]

    x_values: List[float] = []
    coord_values: List[float] = []
    y_values: List[float] = []

    # Evaluate the first endpoint before beginning trapezoidal accumulation.
    x = domain.x_min
    f_last = math.exp(ln_shape_function(x, p, domain))
    y_last = pref * f_last
    jac_last = coordinate_jacobian(quantity, x, T)

    coord_last = (
        x_to_wavelength(x, T)
        if quantity == "wavelength"
        else x_to_frequency(x, T)
    )

    x_values.append(x)
    coord_values.append(coord_last)
    y_values.append(y_last)

    x_peak = x
    f_peak = f_last
    coord_peak = coord_last

    dimensionless_area = 0.0
    physical_integral = 0.0

    for i in range(1, n_steps + 1):
        x = domain.x_min + i * dx
        f = math.exp(ln_shape_function(x, p, domain))
        y = pref * f
        jac = coordinate_jacobian(quantity, x, T)
        coord = (
            x_to_wavelength(x, T)
            if quantity == "wavelength"
            else x_to_frequency(x, T)
        )

        x_values.append(x)
        coord_values.append(coord)
        y_values.append(y)

        if f > f_peak:
            f_peak = f
            x_peak = x
            coord_peak = coord

        # Exactly n_steps trapezoids span the n_steps intervals.
        dimensionless_area += 0.5 * (f_last + f) * dx
        physical_integral += 0.5 * (y_last * jac_last + y * jac) * dx

        f_last = f
        y_last = y
        jac_last = jac

    x_label, y_label = units_label(quantity)

    return Planck2Result(
        quantity=quantity,
        T=T,
        x_values=x_values,
        coord_values=coord_values,
        y_values=y_values,
        x_peak=x_peak,
        coord_peak=coord_peak,
        y_peak=pref * f_peak,
        dimensionless_area=dimensionless_area,
        physical_integral=physical_integral,
        exact_physical_integral=exact_physical_integral(quantity, T),
        physical_integral_units=physical_integral_units(quantity),
        x_label=x_label,
        y_label=y_label,
    )
