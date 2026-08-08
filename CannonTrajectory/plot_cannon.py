"""
plot_cannon.py

Plotting routine for CannonTrajectory.
"""

import matplotlib.pyplot as plt


def plot_cannon(xs, hs):
    """
    Plot the trajectory of the projectile.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(xs, hs, label="Projectile trajectory")
    plt.scatter([0], [0], color="orange", label="Launch point")

    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Vertical distance (m)")
    plt.title("CannonTrajectory — Newtonian Projectile Motion")
    plt.axis("equal")  # Sets plot axes so that "force equal ranges"
    plt.grid(True)
    plt.legend()
    plt.show()
