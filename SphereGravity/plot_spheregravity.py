"""
plot_spheregravity.py

Plotting routine for SphereGravity.
Displays acceleration or relative difference vs radius.
"""

import matplotlib.pyplot as plt
import numpy as np

from physics_spheregravity import OutputType, SHELL_RADIUS


def plot_spheregravity(radius, acceleration, outputType: OutputType = "acceleration"):
    if outputType not in ("acceleration", "relative difference"):
        raise ValueError(
            "outputType must be 'acceleration' or 'relative difference'."
        )

    radius = np.asarray(radius)
    acceleration = np.asarray(acceleration)

    if radius.ndim != 1 or acceleration.ndim != 1:
        raise ValueError("radius and acceleration must be one-dimensional arrays.")
    if radius.size == 0 or acceleration.size == 0:
        raise ValueError("radius and acceleration must not be empty.")
    if radius.size != acceleration.size:
        raise ValueError("radius and acceleration must have the same length.")
    if (
        not np.issubdtype(radius.dtype, np.number)
        or not np.issubdtype(acceleration.dtype, np.number)
        or np.issubdtype(radius.dtype, np.complexfloating)
        or np.issubdtype(acceleration.dtype, np.complexfloating)
    ):
        raise ValueError("radius and acceleration must contain real numeric values.")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(acceleration)):
        raise ValueError("radius and acceleration must contain only finite values.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(radius, acceleration, linewidth=2)

    # Mark the shell boundary at r = 1 with a vertical dashed line
    ax.axvline(
        x=SHELL_RADIUS,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="Shell surface (r = 1)",
    )
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
