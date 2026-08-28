"""
Multiple: Newtonian N-body motion in three dimensions.
"""

import argparse

import physics_multiple
from driver_multiple import SimulationParams, run_simulation
from plot_multiple import animate_multiple, plot_energy_drift, plot_trajectories


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Multiple",
        description="Simulate Newtonian motion for multiple gravitating bodies.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"Multiple {physics_multiple.MODEL_VERSION} "
            f"(build {physics_multiple.BUILD_ID})"
        ),
    )
    return parser.parse_args()


def main():
    parse_args()

    # A Schutz-style planar three-body encounter. The initial positions and
    # velocities all lie in the xy plane; an out-of-plane velocity is a useful
    # later experiment.
    n_bodies = 3
    masses_solar = [1.0, 1.0, 1.0]

    positions_init = [
        [4.6e10, 0.0, 0.0],
        [-4.6e10, 0.0, 0.0],
        [0.0, 4.6e10, 0.0],
    ]

    velocities_init = [
        [0.0, 0.0, 0.0],
        [0.0, -30000.0, 0.0],
        [-30000.0, 0.0, 0.0],
    ]

    params = SimulationParams(
        n_bodies=n_bodies,
        masses_solar=masses_solar,
        positions_init=positions_init,
        velocities_init=velocities_init,

        dt=2000.0,
        max_steps=60000,
        eps1=0.005,
        eps2=1.0e-7,

        # Output: "trajectories" or "animation"
        output_type="animation",

        # Animation controls
        animation_mode="trails",       # "current positions" or "trails"
        frame_time=2.0e5,              # simulated seconds between frames
        frame_interval_ms=50,          # real milliseconds between frames
        trail_time=6.0e5,              # simulated seconds of recent trail
        projection="xy",               # "xy", "xz", or "yz"
        axis_mode="fixed",             # "fixed" or "auto"
    )

    # This optional diagnostic is displayed only in trajectories mode.
    show_energy_diagnostic = False

    result = run_simulation(params)

    print(
        f"Multiple {result['model_version']} (build {result['build_id']}) — "
        f"accepted steps: {result['accepted_steps']}; "
        f"simulated time: {result['final_time'] / 86400.0:.3f} days"
    )
    print(
        "Maximum conservation drift: "
        f"energy={result['max_fractional_energy_drift']:.3e}, "
        f"momentum={result['max_momentum_drift']:.3e}, "
        f"angular momentum={result['max_angular_momentum_drift']:.3e}"
    )

    if result["type"] == "trajectories":
        plot_trajectories(result, projection=params.projection)
        if show_energy_diagnostic:
            plot_energy_drift(result)
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
