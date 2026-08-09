"""
Compute the orbit of a planet around a star in Newton's theory of gravity.
The user can choose the mass of the central star and any desired starting
position and velocity for the planet. The program can output the orbit,
he velocity information, the position or velocity as a function of time,
or the kinetic and potential energies as functions of time. The program
introduces automatic time-step adjustment and the predictor-corrector
technique, both of which improve accuracy.
"""

from driver_orbit import run_orbit
from plot_orbit import plot_orbit


def main():
    # Example initial conditions (user may overwrite)
    xInit = 4.6e10
    yInit = 0.0
    vxInit = 0.0
    vyInit = 58980.0

    k = 1.327e20      # GM for the Sun

    # Time-step and accuracy parameters
    dt0 = 1e4         # initial time-step
    maxSteps = 20000
    eps1 = 0.05       # time-step accuracy threshold
    eps2 = 1e-4       # predictor-corrector accuracy threshold

    # Choose output type (can be changed by user)
    output = "orbit"  # "orbit", "velocity", "position_time", "velocity_time", "energy"

    xs, ys = run_orbit(
        xInit=xInit, yInit=yInit,
        vxInit=vxInit, vyInit=vyInit,
        k=k,
        dt0=dt0,
        maxSteps=maxSteps,
        eps1=eps1,
        eps2=eps2,
        output=output,
    )

    plot_orbit(xs, ys)


if __name__ == "__main__":
    main()
