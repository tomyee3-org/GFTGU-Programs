"""
planck_driver.py

Faithful Python port of Bernard Schutz's Planck program (Gravity From the Ground Up),
Creative Commons BY‑NC‑SA 1.0.

Contains the numerical driver that:
    - steps through x
    - computes Planck function values
    - finds the peak location
    - computes the area under the curve
"""

from planck_physics import planck_small_x, planck_large_x, planck_exact


def run_planck(n_steps: int, output_type: str):
    """
    Executes Schutz's Planck algorithm.

    Parameters
    ----------
    n_steps : int
        Number of divisions of the x-domain.
    output_type : str
        "maximum" or "area"

    Returns
    -------
    dict
        Contains either:
            {"x_peak": value, "y_peak": value}
        or
            {"area": value}
    """

    # ------------------------------------------------------------------
    # Domain and constants (exactly as in Schutz's Java)
    # ------------------------------------------------------------------
    x_min = 0.01
    x_max = 100.0
    x_low = 0.05
    x_high = 20.0

    dx = (x_max - x_min) / n_steps

    area = 0.0
    x_peak = 0.0
    y_peak = 0.0
    y_last = 0.0

    # ------------------------------------------------------------------
    # Main loop over x (faithful to Schutz)
    # ------------------------------------------------------------------
    x = x_min
    for _ in range(n_steps + 1):

        # Compute Planck function using correct regime
        if x < x_low:
            y = planck_small_x(x)
        elif x > x_high:
            y = planck_large_x(x)
        else:
            y = planck_exact(x)

        # Peak detection (Wien's law)
        if y > y_peak:
            y_peak = y
            x_peak = x

        # Area under x^3/(exp(x)-1) curve → divide by x^2
        y_area = y / (x * x)
        area += 0.5 * (y_area + y_last) * dx
        y_last = y_area

        x += dx

    # ------------------------------------------------------------------
    # Output selection
    # ------------------------------------------------------------------
    if output_type.lower() == "maximum":
        return {"x_peak": x_peak, "y_peak": y_peak}
    else:
        return {"area": area}
