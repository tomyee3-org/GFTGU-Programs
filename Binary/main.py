"""
Binary: Newtonian motion of two mutually gravitating point masses.

Edit the parameters below to explore circular, elliptical, parabolic,
and hyperbolic two-body trajectories.
"""

from driver_binary import integrate_binary
from plot_binary import OutputType, plot_binary


def main():
    MA = 2.0e30
    MB = 2.0e30

    xInitA = 4.6e10
    yInitA = 0.0
    xInitB = -4.6e10
    yInitB = 0.0

    vInitA = 0.0
    uInitA = 13000.0
    vInitB = 0.0
    uInitB = -13000.0

    dt = 2000
    max_steps = 10000
    eps1 = 0.05
    eps2 = 1.0e-4

    # True: stop when the relative position completes one revolution.
    # False: run to max_steps; useful for unbound or long-term energy tests.
    stop_after_one_orbit = True

    # "orbits"
    # "velocity space"
    # "position vs. time, body A"
    # "position vs. time, body B"
    # "velocity vs. time, body A"
    # "velocity vs. time, body B"
    # "energy vs time"
    output_type: OutputType = "orbits"

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
        stop_after_one_orbit=stop_after_one_orbit,
    )

    plot_binary(result, output_type)


if __name__ == "__main__":
    main()
