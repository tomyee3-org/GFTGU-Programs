"""
Plotting routine for CannonTrajectory.
"""

import matplotlib.pyplot as plt


def plot_cannon(xs, hs):
    """Plot the projectile trajectory."""
    return plot_cannon_overlay([("Projectile trajectory", xs, hs)])


def _unpack_trajectory(item):
    """Return ``(label, xs, hs)`` from one overlay item or raise ValueError.

    A string is refused explicitly: a three-character string would otherwise
    unpack into three one-character values and be plotted as categories.
    """
    if not isinstance(item, (str, bytes)):
        try:
            label, xs, hs = item
        except (TypeError, ValueError):
            pass
        else:
            return label, xs, hs
    raise ValueError(
        "each trajectory must be a (label, xs, hs) triple, for example "
        '("45 degrees", xs, hs)'
    )


def plot_cannon_overlay(trajectories):
    """Plot labeled trajectories together on one figure.

    ``trajectories`` is an iterable of ``(label, xs, hs)`` triples.  This
    compact interface keeps the plotting details out of parameter-sweep
    exercises while leaving students responsible for generating the data.
    Any item that is not exactly such a triple raises ``ValueError``.  Code
    that extends this helper must keep accepting that triple form.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        count = 0
        for item in trajectories:
            label, xs, hs = _unpack_trajectory(item)
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
