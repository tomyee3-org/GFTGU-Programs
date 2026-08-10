"""
Compute the orbit of a planet around a star in Newton's theory of gravity.
The user can choose the mass of the central star and any desired starting
position and velocity for the planet. The program can output the orbit,
the velocity information, the position or velocity as a function of time,
or the kinetic and potential energies as functions of time. The program
introduces automatic time-step adjustment and the predictor-corrector
technique, both of which improve accuracy.
"""

from driver_orbit import run_orbit, OutputType
from plot_orbit import plot_orbit


def main():
    # Example initial conditions (user may overwrite): Mercury's orbit
    # around the Sun, starting at perihelion.
    xInit = 4.6e10       # Mercury's perihelion distance (m)
    yInit = 0.0
    vxInit = 0.0
    vyInit = 58980.0      # Mercury's perihelion speed (m/s)

    # k = GM for the Sun.
    # The directly-measured product of G and mass of the Sun is known
    # far more precisely than G and M_sun separately
    k = 1.327e20

    # Time-step and accuracy parameters
    dt0 = 1e4         # initial time-step
    maxSteps = 20000
    eps1 = 0.05       # time-step accuracy threshold
    eps2 = 1e-4       # predictor-corrector accuracy threshold

    # Choose what to plot -- this only affects plot_orbit() below, not
    # the computation itself, since run_orbit() always computes the
    # full position/velocity/time/energy history regardless of which
    # of these you pick:
    #   "orbit"          -- the x-y trajectory (a closed ellipse for a
    #                        bound orbit)
    #   "velocity"       -- velocity-space "hodograph", i.e. the curve
    #                        traced by (vx, vy) -- always a circle for
    #                        1/r^2 gravity, regardless of how eccentric
    #                        the orbit itself is
    #   "position_time"  -- x(t) and y(t) together, to read off the
    #                        orbital period
    #   "velocity_time"  -- vx(t) and vy(t) together
    #   "energy"         -- specific kinetic/potential/total energy vs.
    #                        time; the total should stay flat, which is
    #                        a direct visual check of energy
    #                        conservation (see Chapter 6)
    output: OutputType = "orbit"

    result = run_orbit(
        xInit=xInit, yInit=yInit,
        vxInit=vxInit, vyInit=vyInit,
        k=k,
        dt0=dt0,
        maxSteps=maxSteps,
        eps1=eps1,
        eps2=eps2,
    )

    plot_orbit(result, output=output)


if __name__ == "__main__":
    main()
