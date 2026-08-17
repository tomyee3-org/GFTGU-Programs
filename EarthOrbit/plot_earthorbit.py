"""
plot_earthorbit.py

Plotting routine for EarthOrbit.
Displays both the Earth's surface and the projectile trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_earth_orbit(xs, ys, xEarth, yEarth):
    """
    Plot the Earth and the projectile trajectory.
    Uses equal axis scaling.
    """
    if len(xs) == 0 or len(xs) != len(ys):
        raise ValueError("xs and ys must have the same nonzero length")
    if len(xEarth) == 0 or len(xEarth) != len(yEarth):
        raise ValueError("xEarth and yEarth must have the same nonzero length")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))
            and np.all(np.isfinite(xEarth)) and np.all(np.isfinite(yEarth))):
        raise ValueError("all plotted coordinate values must be finite")

    plt.figure(figsize=(8, 8))

    # Earth surface
    plt.plot(xEarth, yEarth, linestyle="--", color="blue",
             label="Earth surface")

    # Trajectory
    plt.plot(xs, ys, color="red", linewidth=2,
             label="Projectile trajectory")

    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title("EarthOrbit — Attempting to Achieve Orbit")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
