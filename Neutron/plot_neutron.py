# plot_neutron.py
"""
Plot module for Neutron star structure.
"""

import matplotlib.pyplot as plt

def plot_neutron(data, output_type):
    r = data["radius"]

    if output_type == "Pressure":
        y = data["pressure"]
        ylabel = "Pressure (Pa)"
    elif output_type == "Density":
        y = data["density"]
        ylabel = "Density (kg/m^3)"
    elif output_type == "Mass":
        y = data["mass"]
        ylabel = "Mass interior (kg)"
    else:
        raise ValueError("Unknown output_type")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, y, lw=2)
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Neutron Star: {output_type}")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
