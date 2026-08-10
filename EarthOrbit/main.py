"""
Find out how to launch a projectile from near the Earth's surface and get it into orbit
around the Earth. The program shows how sensitive the trajectory is to the initial speed
of the projectile.
"""

from driver_earthorbit import run_earth_orbit
from plot_earthorbit import plot_earth_orbit

# The "simplified" (default) force_law assumes that
# gravity has constant magnitude g, only its direction changes with
# position. This is a good approximation near the surface but visibly
# breaks down at higher launch altitudes (e.g. a precessing, non-closing
# rosette instead of a clean ellipse) -- which is the point of this
# exercise. Set force_law="inverse_square" to use the physically
# accurate law instead, for comparison.

force_law = "simplified"

xs, ys, xEarth, yEarth = run_earth_orbit(
    h0=300.0,
    uInit=7900.0,  # Not quite sufficient velocity to complete an orbit at 300.0 meters
    vInit=0.0,
    dt=0.4,
    maxSteps=15000,
    force_law=force_law,
)

plot_earth_orbit(xs, ys, xEarth, yEarth)
