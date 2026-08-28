"""
Plotting for Neutron star structure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physics_neutron import M_SUN


_VALID_OUTPUTS = {"Pressure", "Density", "Mass"}


def plot_neutron(data: dict, output_type: str, *, log_y: bool = False) -> None:
    """
    Plot pressure, density, or enclosed mass versus radius.

    Radius is shown in km. Mass is shown in solar masses. Pressure and density
    may optionally use a logarithmic y-axis.
    """
    if output_type not in _VALID_OUTPUTS:
        raise ValueError(
            'output_type must be exactly "Pressure", "Density", or "Mass".'
        )
    if not isinstance(log_y, bool):
        raise ValueError("log_y must be True or False.")

    r_km = np.asarray(data["radius"], dtype=float) / 1000.0

    if output_type == "Pressure":
        y = np.asarray(data["pressure"], dtype=float)
        ylabel = "Pressure (Pa)"
    elif output_type == "Density":
        y = np.asarray(data["density"], dtype=float)
        ylabel = r"Density (kg/m$^3$)"
    else:
        y = np.asarray(data["mass"], dtype=float) / M_SUN
        ylabel = r"Enclosed mass ($M_\odot$)"

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_y and output_type in {"Pressure", "Density"}:
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
    ax.set_title(f"Neutron Star: {output_type}")
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
