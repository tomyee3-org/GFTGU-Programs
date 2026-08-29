"""User entry point for Planck2."""

import argparse
import math

import planck2_physics
from planck2_driver import run_planck2
from planck2_plot import Corner, plot_planck2
from planck2_physics import PlanckDomain, PlanckQuantity, SHAPE_EXPONENT


def _exact_dimensionless_area(p: int) -> float:
    """Exact integral from 0 to infinity of x^p/(exp(x)-1) dx for p=3 or 5."""
    if p == 3:
        return math.pi**4 / 15.0
    if p == 5:
        return 8.0 * math.pi**6 / 63.0
    raise ValueError(f"No closed form on hand for p={p}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Planck2",
        description="Explore black-body spectra in dimensionless and SI forms.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Planck2 {planck2_physics.MODEL_VERSION} "
            f"(build {planck2_physics.BUILD_ID})"
        ),
    )
    return parser.parse_args()


def main():
    parse_args()

    # ----------------------------------------------------------------
    # User-editable parameters
    # ----------------------------------------------------------------
    T = 5900.0
    quantity: PlanckQuantity = "wavelength"  # wavelength, frequency, energy_density
    n_steps = 2000

    # Dimensionless x-domain and approximation boundaries.
    x_min = 0.01
    x_max = 100.0
    x_low = 0.05
    x_high = 20.0

    # Plot controls.
    corner: Corner = "upper right"
    y_frac_window = 0.003   # 0 shows the entire physical-coordinate domain

    domain = PlanckDomain(
        x_min=x_min,
        x_max=x_max,
        x_low=x_low,
        x_high=x_high,
    )

    result = run_planck2(
        T=T,
        quantity=quantity,
        n_steps=n_steps,
        domain=domain,
    )
    plot_planck2(
        result,
        corner=corner,
        y_frac_window=y_frac_window,
    )

    exact_shape = _exact_dimensionless_area(SHAPE_EXPONENT[quantity])
    print(
        f"Planck2 {result.model_version} (build {result.build_id}) — "
        f"peak at x = {result.x_peak:.6f}"
    )
    print(
        f"Dimensionless area = {result.dimensionless_area:.6f} "
        f"(0..infinity exact value: {exact_shape:.6f})"
    )
    print(
        f"Physical integral = {result.physical_integral:.6e} "
        f"{result.physical_integral_units}"
    )
    print(
        f"Exact 0..infinity bolometric value = {result.exact_physical_integral:.6e} "
        f"{result.physical_integral_units}"
    )


if __name__ == "__main__":
    main()
