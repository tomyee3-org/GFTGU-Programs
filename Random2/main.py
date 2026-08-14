"""
main.py

User entry point for Random2, offering a choice of two displays:

    "scaled_distance" -- average scaled walk
        distance vs. number of steps, on log-log axes, demonstrating
        the sqrt(N) diffusion law.

    "walk2d" -- a handful of individual 2D random walks
        drawn from the center of a circular "star" outward, each in
        its own color, with walks that reach the surface continuing as
        a straight escaping ray.

Example parameters are provided below and may be overwritten by the
user.
"""

from random2_driver import run_scaled_distance_experiment, run_walk2d
from random2_plot import plot_scaled_distance, plot_walk2d


def main():
    # ----------------------------------------------------------------
    # Choose which display to run
    # ----------------------------------------------------------------
    display = "scaled_distance"   # "scaled_distance" or "walk2d"

    if display == "scaled_distance":
        maxSteps = 4096
        nTrials = 100
        lengths, avg_dist = run_scaled_distance_experiment(maxSteps, nTrials)
        plot_scaled_distance(lengths, avg_dist)

    else:  # "walk2d"
        reference_steps = 2000  # the "N" the default radius is sized around
        n_walks = 4              # number of simultaneous walks to draw
        mean_free_path = 1.0     # used only for the default radius
        radius_factor = 2.0      # default radius = radius_factor * sqrt(reference_steps * mean_free_path)
        radius = None            # or set an explicit radius to override the default
        corner = "upper right"

        result = run_walk2d(
            reference_steps=reference_steps,
            n_walks=n_walks,
            radius=radius,
            mean_free_path=mean_free_path,
            radius_factor=radius_factor,
        )
        plot_walk2d(result, corner=corner)


if __name__ == "__main__":
    main()
