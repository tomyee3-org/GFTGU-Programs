"""
plot_cannon.py

Plotting routine for CannonTrajectory.
"""

import matplotlib.pyplot as plt


def plot_cannon(xs, hs):
    """
    Plot the trajectory of the projectile.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, hs, label="Projectile trajectory")
    ax.scatter([0], [0], color="orange", label="Launch point")

    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Vertical distance (m)")
    ax.set_title("CannonTrajectory — Newtonian Projectile Motion")
    ax.set_aspect("equal", "box")  # Sets plot axes so that "force equal ranges"
    ax.grid(True)
    ax.legend()
    plt.show()
