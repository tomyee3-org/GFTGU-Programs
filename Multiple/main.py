"""Multiple: Newtonian N-body motion in three dimensions."""

from driver_multiple import SimulationParams, run_simulation
from plot_multiple import animate_multiple, plot_trajectories


def main():
    n_bodies = 3
    masses_solar = [1.0, 1.0, 1.0]

    positions_init = [
        [4.6e10, 0.0, 0.0],
        [-4.6e10, 0.0, 0.0],
        [0.0, 4.6e10, 0.0],
    ]
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
        dt=2000.0,
        max_steps=40000,
        eps1=0.05,
        eps2=0.0001,

        output_type="animation",          # "trajectories" or "animation"
        animation_mode="trails",          # "current positions" or "trails"
        frame_time=2.0e5,                 # simulated seconds between frames
        frame_interval_ms=50,             # real milliseconds between frames
        trail_time=6.0e5,                 # simulated seconds of recent trail
        projection="xy",                  # "xy", "xz", or "yz"
        axis_mode="fixed",                # "fixed" or "auto"
    )

    result = run_simulation(params)

    if result["type"] == "trajectories":
        plot_trajectories(result, projection=params.projection)
    else:
        n_frames = len(result["frame_times"])
        playback_seconds = n_frames * params.frame_interval_ms / 1000.0
        print(
            f"Animation frames: {n_frames}; "
            f"frame spacing: {params.frame_time:.4g} simulated s; "
            f"approximate playback time: {playback_seconds:.1f} real s"
        )
        animate_multiple(result)


if __name__ == "__main__":
    main()
