"""
plot_spheregravity.py

Plotting routine for SphereGravity.
Displays acceleration or relative difference vs radius.
"""

import matplotlib.pyplot as plt

from physics_spheregravity import OutputType


def plot_spheregravity(radius, acceleration, outputType: OutputType = "acceleration"):
    if outputType not in ("acceleration", "relative difference"):
        raise ValueError(
            "outputType must be 'acceleration' or 'relative difference'."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(radius, acceleration, linewidth=2)

    # Mark the shell boundary at r = 1 with a vertical dashed line
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2.0, label='Shell surface (r = 1)')
    ax.legend(fontsize=9)

    ax.set_xlabel("Radius r  (shell radius = 1)")
    if outputType == "acceleration":
        ax.set_ylabel("Gravitational acceleration (dimensionless)")
        ax.set_title("Gravitational Acceleration of a Thin Spherical Shell")
    else:
        ax.set_ylabel("Relative difference from Newton's theorem")
        ax.set_title("Relative Difference from Newton's Shell Theorem")

    ax.grid(True)
    plt.tight_layout()
    plt.show()
