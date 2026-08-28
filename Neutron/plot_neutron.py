"""
Plotting for Neutron star structure.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from physics_neutron import M_SUN


_VALID_OUTPUTS = {"Pressure", "Density", "Mass"}
_OUTPUT_ALIASES = {name.casefold(): name for name in _VALID_OUTPUTS}


def plot_neutron(data: dict, output_type: str, *, log_y: bool = False) -> None:
    """
    Plot pressure, density, or enclosed mass versus radius.

    Radius is shown in km. Mass is shown in solar masses. Pressure and density
    may optionally use a logarithmic y-axis.
    """
    if not isinstance(output_type, str):
        raise ValueError(
            'output_type must be "Pressure", "Density", or "Mass".'
        )
    canonical_output = _OUTPUT_ALIASES.get(output_type.strip().casefold())
    if canonical_output is None:
        raise ValueError(
            'output_type must be "Pressure", "Density", or "Mass" '
            "(capitalization is ignored)."
        )
    if not isinstance(log_y, bool):
        raise ValueError("log_y must be True or False.")
    if canonical_output == "Mass" and log_y:
        warnings.warn(
            "log_y=True applies only to Pressure and Density; the Mass "
            "profile will use a linear y-axis.",
            UserWarning,
            stacklevel=2,
        )

    r_km = np.asarray(data["radius"], dtype=float) / 1000.0

    if canonical_output == "Pressure":
        y = np.asarray(data["pressure"], dtype=float)
        ylabel = "Pressure (Pa)"
    elif canonical_output == "Density":
        y = np.asarray(data["density"], dtype=float)
        ylabel = r"Density (kg/m$^3$)"
    else:
        y = np.asarray(data["mass"], dtype=float) / M_SUN
        ylabel = r"Enclosed mass ($M_\odot$)"

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_y and canonical_output in {"Pressure", "Density"}:
        # A logarithmic axis cannot display the exact zero-valued surface point.
        # Mask nonpositive values explicitly rather than relying on Matplotlib to
        # discard them.
        positive = y > 0.0
        ax.plot(r_km[positive], y[positive], lw=2)
        ax.set_yscale("log")
    else:
        ax.plot(r_km, y, lw=2)

    ax.set_xlabel("Radius (km)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Neutron Star: {canonical_output}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_model_summary(data: dict) -> None:
    """Print the most useful global properties of the computed model."""
    print("Neutron-star model summary")
    print(f"  surface radius : {data['surface_radius_km']:.3f} km")
    print(f"  total mass     : {data['total_mass_solar']:.6f} M_sun")
    print(f"  central density: {data['rhoC']:.6e} kg/m^3")
    print(
        "  central sound  : "
        f"v_s/c = {data['central_sound_speed_over_c']:.6f}"
    )
    print(f"  compactness    : 2GM/(Rc^2) = {data['compactness']:.6f}")
    print(
        "  causality check: "
        + ("satisfied" if data["causality_satisfied"] else "NOT satisfied")
    )
    print(
        "  Buchdahl check : "
        + ("satisfied" if data["buchdahl_satisfied"] else "NOT satisfied")
    )
    print(
        f"  radial samples : {len(data['radius'])} "
        f"(nominal dr={data['dr_nominal']:.3f} m)"
    )
