"""
Investigate the most elementary problem of motion in gravity:
the trajectory of a cannonball near the surface of the Earth.
"""

from driver_cannon import run_cannon_trajectory
from plot_cannon import plot_cannon

xs, hs = run_cannon_trajectory(speed=100, angle_deg=45, dt=0.1,
                               method="improved")

plot_cannon(xs, hs)
