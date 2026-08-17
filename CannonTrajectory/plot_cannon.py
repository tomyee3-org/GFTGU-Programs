"""
Plotting routine for CannonTrajectory.
"""

import matplotlib.pyplot as plt


def plot_cannon(xs, hs):
    """Plot the projectile trajectory."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, hs, label="Projectile trajectory")
    ax.scatter([0], [0], color="orange", label="Launch point")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Vertical distance (m)")
    ax.set_title("CannonTrajectory — Newtonian Projectile Motion")
    ax.set_aspect("equal", "box")  # Same physical scale on both axes.
    ax.grid(True)
    ax.legend()
    plt.show()
