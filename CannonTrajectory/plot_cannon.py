"""
Plotting routine for CannonTrajectory.
"""

import matplotlib.pyplot as plt


def plot_cannon(xs, hs):
    """Plot the projectile trajectory."""
    return plot_cannon_overlay([("Projectile trajectory", xs, hs)])


def plot_cannon_overlay(trajectories):
    """Plot labeled trajectories together on one figure.

    ``trajectories`` is an iterable of ``(label, xs, hs)`` triples.  This
    compact interface keeps the plotting details out of parameter-sweep
    exercises while leaving students responsible for generating the data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        count = 0
        for label, xs, hs in trajectories:
            ax.plot(xs, hs, label=label)
            count += 1

        if count == 0:
            raise ValueError("at least one trajectory is required")

        ax.scatter([0], [0], color="orange", label="Launch point")
        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Vertical distance (m)")
        ax.set_title("CannonTrajectory — Newtonian Projectile Motion")
        ax.set_aspect("equal", "box")  # Same physical scale on both axes.
        ax.grid(True)
        ax.legend()
        plt.show()
    except Exception:
        plt.close(fig)
        raise

    return fig, ax
