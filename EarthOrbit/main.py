from driver_earthorbit import run_earth_orbit
from plot_earthorbit import plot_earth_orbit

xs, ys, xEarth, yEarth = run_earth_orbit(   # lowercase xs — fix
    h0=300000.0,
    uInit=9000.0,
    vInit=0.0,
    dt=0.4,
    maxSteps=50000
)

plot_earth_orbit(xs, ys, xEarth, yEarth)
