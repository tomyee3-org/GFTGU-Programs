"""
main.py

User entry point for Planck2, offering a choice of three Planck-law
quantities (spectral radiance per unit wavelength, spectral radiance
per unit frequency, or spectral energy density per unit frequency),
plotted against physical units with the peak location and area
annotated directly on the graph.

Example parameters are provided below and may be overwritten by the
user.
"""

import math

from planck2_driver import run_planck2
from planck2_plot import plot_planck2, Corner
from planck2_physics import SHAPE_EXPONENT, PlanckQuantity


def _exact_area(p: int) -> float:
    """
    Exact closed-form value of the integral(0 to infinity) of
    x^p/(e^x-1) dx = p! * zeta(p+1), specialized for the two integer
    exponents this program uses (p=3 and p=5), each of which reduces
    to a clean multiple of a power of pi:
        p=3: 3! * zeta(4) = 6 * (pi^4/90)  = pi^4/15    (Investigation 10.3)
        p=5: 5! * zeta(6) = 120 * (pi^6/945) = 8*pi^6/63
    """
    if p == 3:
        return math.pi**4 / 15.0
    elif p == 5:
        return 8.0 * math.pi**6 / 63.0
    else:
        raise ValueError(f"No closed form on hand for p={p}")


def main():
    # ----------------------------------------------------------------
    # Example parameters (user may overwrite)
    # ----------------------------------------------------------------
    T = 5900.0                    # temperature in kelvin (e.g. the Sun)
    quantity: PlanckQuantity = "wavelength"  # "wavelength", "frequency", or "energy_density"
    n_steps = 2000                # resolution of the dimensionless domain
    corner: Corner = "upper right"           # where to place the results annotation

    # ----------------------------------------------------------------
    # Run and plot
    # ----------------------------------------------------------------
    result = run_planck2(T=T, quantity=quantity, n_steps=n_steps)
    plot_planck2(result, corner=corner)

    exact = _exact_area(SHAPE_EXPONENT[quantity])
    print(f"Peak at x = {result.x_peak:.6f}")
    print(f"Area under f(x) = {result.area:.6f}  (true value: {exact:.6f})")


if __name__ == "__main__":
    main()
