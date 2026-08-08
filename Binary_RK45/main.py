"""
Main entry point for binary orbit simulation using RK45
rather than the simple predictor–corrector scheme and
adaptive time-step logic employed by module Binary
"""

from driver_binary import run_binary_orbit
from plot_binary import plot_orbits

def main():
    # Example initial conditions (user may overwrite)
    MA = 2.0e30      # kg (roughly solar mass)
    MB = 2.0e30      # kg

    # Initial positions (meters)
    xInitA = 4.6e10
    yInitA = 0.0
    xInitB =  -4.6e10
    yInitB = 0.0

    # Initial velocities (m/s)
    vInitA = 0.0
    uInitA = 13000.0
    vInitB = 0.0
    uInitB = -13000.0

    # Integration parameters
    t_max = 1e9      # seconds
    rtol = 1e-9
    atol = 1e-9

    data = run_binary_orbit(
        MA, MB,
        xInitA, yInitA, xInitB, yInitB,
        vInitA, uInitA, vInitB, uInitB,
        t_max=t_max,
        rtol=rtol,
        atol=atol
    )

    plot_orbits(data)

if __name__ == "__main__":
    main()
