"""
Find out how to launch a projectile from near Earth's surface and get it into
orbit. The program shows how sensitive the trajectory is to the initial speed.

The default "simplified" force law gives gravity a constant magnitude g,
while its direction always points toward Earth's centre.
Set force_law="inverse_square" for higher-altitude and Keplerian experiments.
"""

from driver_earthorbit import run_earth_orbit
from plot_earthorbit import plot_earth_orbit

force_law = "simplified"

xs, ys, xEarth, yEarth = run_earth_orbit(
    h0=300.0,
    uInit=7900.0,  # tangential/horizontal; this default is slightly sub-orbital
    vInit=0.0,     # radial/vertical
    dt=0.4,
    maxSteps=15000,
    force_law=force_law,
)

plot_earth_orbit(xs, ys, xEarth, yEarth)
