"""
plot_spheregravity.py

Plotting routine for SphereGravity.
Displays acceleration or relative difference vs radius.
"""

import matplotlib.pyplot as plt

from physics_spheregravity import OutputType


def plot_spheregravity(radius, acceleration, outputType: OutputType = "acceleration"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(radius, acceleration, linewidth=2)

    ax.set_xlabel("Radius r")
    if outputType == "acceleration":
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Gravitational Acceleration of Thin Spherical Shell")
    else:
        ax.set_ylabel("Relative Difference")
        ax.set_title("Relative Difference from Newton's Theorem")

    ax.grid(True)
    plt.show()
