"""
main.py

User entry point for Random2.

Choose one display:

    "scaled_distance"
        Statistical random-walk experiment showing the sqrt(N)
        diffusion law.

    "walk2d"
        A small number of fixed-step, isotropic 2D walks drawn inside
        a circular schematic star.
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
        step_distribution = "uniform"   # "uniform" or "gaussian"

        lengths, avg_dist = run_scaled_distance_experiment(
            maxSteps,
            nTrials,
            step_distribution=step_distribution,
        )
        plot_scaled_distance(lengths, avg_dist)

    elif display == "walk2d":
        reference_steps = 2000
        n_walks = 4
        mean_free_path = 1.0
        radius_factor = 2.0
        radius = None
        ray_length_factor = 0.6
        step_cap = 200_000
        corner = "upper right"

        result = run_walk2d(
            reference_steps=reference_steps,
            n_walks=n_walks,
            radius=radius,
            mean_free_path=mean_free_path,
            radius_factor=radius_factor,
            ray_length_factor=ray_length_factor,
            step_cap=step_cap,
        )
        plot_walk2d(result, corner=corner)

    else:
        raise ValueError(
            'display must be "scaled_distance" or "walk2d".'
        )


if __name__ == "__main__":
    main()
