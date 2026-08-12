"""
MercPert main module

Simulate the motion of a small planet in a solar system consisting of a star
like the Sun and a large planet more massive than Jupiter and closer to the Sun.
The program illustrates the way that massive planets "clean out" regions of the
planetary system near themselves. Spectacular interactions between the small
planet and the massive one are easy to achieve with the suggested initial data.

Example entry point for running the MercPert simulation
and plotting the orbits.

All parameters here are examples that users can overwrite.
"""

from physics_mercpert import (
    BinarySystemParams,
    MercuryInitialConditions,
    AU,
)
from driver_mercpert import MercPertRunParams, run_mercpert
from plot_mercpert import plot_orbits


def main():
    # Example binary system: Sun + super-Jupiter closer than Jupiter,
    # roughly at Venus's distance (~0.7 AU).
    binary_params = BinarySystemParams(
        m_sun_solar=1.0,            # Sun-like central star
        m_planet_solar=0.1,         # ~100 times Jupiter's mass
        binary_separation=0.7 * AU, # Note that the Jupiter orbits the BARYCENTER of the Sun-Jupiter system
                                    # so that, given the default mass ratio, the radius of the green circle
                                    # will be 0.6364 AU
    )

    # Example initial conditions for "Mercury"
    # These are illustrative; users can adjust to reproduce
    # the book's figures or explore chaotic behavior.
    merc_ic = MercuryInitialConditions(
        x_init=0.3 * AU,         # start outside the binary
        y_init=0.0,
        vx_init=0.0,
        vy_init=59220.0,         # some tangential velocity (m/s)
    )

    # Driver parameters
    run_params = MercPertRunParams(
        dt=2000.0,       # initial time-step (s), as in examples
        max_steps=400000,
        eps1=0.05,       # time-step halving threshold
        eps2=0.0001,       # predictor-corrector convergence threshold
    )

    output = run_mercpert(binary_params, merc_ic, run_params)
    plot_orbits(output, title="MercPert orbits", merc_ic=merc_ic)


if __name__ == "__main__":
    main()
