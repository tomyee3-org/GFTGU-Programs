"""
Main entry point for Binary orbit simulation.
Simulate the motion of two stars in a binary orbit around one another.
The user can choose initial stellar masses, positions and velocities to
represent any kind of orbit. The program can output graphs of the orbits,
of the velocity evolution, of the positions and speeds as functions of
time, and of the kinetic and potential energies as functions of time.

Example parameters are set to values that roughly produce an elliptical
binary orbit of two equal-mass stars.

Users can overwrite these values as desired.
"""

from driver_binary import integrate_binary
from plot_binary import plot_binary, OutputType


def main():
    # Example initial conditions (SI units)
    # Two equal-mass bodies, symmetric positions and velocities.
    MA = 2.0e30  # kg
    MB = 2.0e30  # kg

    # Positions (meters)
    xInitA = 4.6e10
    yInitA = 0.0
    xInitB = -4.6e10
    yInitB = 0.0

    # Velocities (m/s) chosen to give a bound, non-circular orbit
    vInitA = 0.0
    uInitA = 13000.0
    vInitB = 0.0
    uInitB = -13000.0

    # Time-step and accuracy parameters
    dt = 2000          # base time-step (s)
    max_steps = 10000    # maximum number of steps
    eps1 = 0.05       # time-step accuracy threshold
    eps2 = 1.0e-4       # predictor–corrector accuracy threshold

    # Choose output type (can be changed by user)
    # Available output_types:
    #   "orbits",
    #   "velocity space",
    #   "position vs. time, body A",
    #   "position vs. time, body B",
    #   "velocity vs. time, body A",
    #   "velocity vs. time, body B",
    #   "energy vs time"

    output_type: OutputType = "velocity space"

    result = integrate_binary(
        MA=MA,
        MB=MB,
        xInitA=xInitA,
        yInitA=yInitA,
        vInitA=vInitA,
        uInitA=uInitA,
        xInitB=xInitB,
        yInitB=yInitB,
        vInitB=vInitB,
        uInitB=uInitB,
        dt=dt,
        max_steps=max_steps,
        eps1=eps1,
        eps2=eps2,
    )

    plot_binary(result, output_type)


if __name__ == "__main__":
    main()
