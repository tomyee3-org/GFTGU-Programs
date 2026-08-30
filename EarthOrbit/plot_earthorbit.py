"""
plot_earthorbit.py

Plotting routine for EarthOrbit.
Displays both the Earth's surface and the projectile trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt


def _validated_curve(x_values, y_values, name):
    """Return two finite, one-dimensional float arrays for plotting."""
    try:
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be numeric") from exc

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError(f"{name} coordinates must be one-dimensional")
    if x_array.size == 0 or x_array.size != y_array.size:
        raise ValueError(
            f"{name} x and y coordinates must have the same nonzero length"
        )
    if not (np.all(np.isfinite(x_array)) and np.all(np.isfinite(y_array))):
        raise ValueError(f"{name} coordinates must all be finite")

    return x_array, y_array


def plot_earth_orbit(xs, ys, xEarth, yEarth):
    """
    Plot the Earth and the projectile trajectory.
    Uses equal axis scaling and returns the matplotlib Figure and Axes.
    """
    xs, ys = _validated_curve(xs, ys, "trajectory")
    xEarth, yEarth = _validated_curve(xEarth, yEarth, "Earth-surface")

    figure, axes = plt.subplots(figsize=(8, 8))

    # Earth surface
    axes.plot(xEarth, yEarth, linestyle="--", color="blue",
              label="Earth surface")

    # Trajectory
    axes.plot(xs, ys, color="red", linewidth=2,
              label="Projectile trajectory")

    axes.set_xlabel("x (meters)")
    axes.set_ylabel("y (meters)")
    axes.set_title("EarthOrbit — Attempting to Achieve Orbit")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True)
    axes.legend()
    plt.show()
    return figure, axes
