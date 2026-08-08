"""
A program to simulate the motions of any number of bodies that interact with
one another gravitationally, within Newton's theory of gravity. The user can set
up fully three-dimensional initial conditions and masses for any number of
objects, and the output can be a drawing of all the trajectories or a continuously
running movie of the locations of the bodies.

This is the main main entry point which sets example parameters (user can edit)
and runs the simulation.
"""

from driver_multiple import SimulationParams, run_simulation
from plot_multiple import plot_trajectories, plot_current_positions


def main():
    # Example: three-body encounter
    n_bodies = 3

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

    params = SimulationParams(
        n_bodies=n_bodies,
        masses_solar=masses_solar,
        positions_init=positions_init,
        velocities_init=velocities_init,
        dt=2000.0,          # initial time-step (s), similar to MercPert defaults
        max_steps=40000,  # user can reduce for testing
        out_steps=1000,    # steps between output events
        output_type="trajectories",  # or "current positions"
        eps1=0.05,       # time-step halving accuracy
        eps2=0.0001,       # predictor-corrector accuracy
    )

    result = run_simulation(params)

    if result["type"] == "trajectories":
        plot_trajectories(result)
    else:
        plot_current_positions(result)


if __name__ == "__main__":
    main()
