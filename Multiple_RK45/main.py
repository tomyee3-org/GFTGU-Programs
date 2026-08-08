# main.py
"""
A program to simulate the motions of any number of bodies that interact with
one another gravitationally, within Newton's theory of gravity. The user can set
up fully three-dimensional initial conditions and masses for any number of
objects, and the output can be a drawing of all the trajectories or a continuously
running movie of the locations of the bodies.

Main entry point for Multiple_RK45.
User sets parameters here.
"""

from driver_multiple import run_simulation
from plot_multiple import plot_trajectories, plot_positions


if __name__ == "__main__":

    # -----------------------------
    # Example initial conditions
    # (faithful to Schutz’s defaults)
    # -----------------------------

    masses_solar = [1.0, 1.0, 1.0]  # all solar-mass bodies

    # Positions (x, y, z) in meters
    positions_init = [
        [4.6e10, 0.0, 0.0],
        [-4.6e10, 0.0, 0.0],
        [0.0, 4.6e10, 0.0],
    ]

    # Velocities (vx, vy, vz) in m/s
    velocities_init = [
        [0.0, 0.0, -30000.0],
        [0.0, -30000.0, 0.0],
        [-30000.0, 0.0, 0.0],
    ]

    # Simulation parameters
    t_max = 3.0e7        # total time (seconds)
    dt_output = 2.0e4    # spacing between output samples

    # Run simulation
    times, trajectories = run_simulation(
        masses_solar, positions_init, velocities_init,
        t_max, dt_output
    )

    # Plot trajectories
    labels = ["Body 1", "Body 2", "Body 3"]
    plot_trajectories(trajectories, labels)
