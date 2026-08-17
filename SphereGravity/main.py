"""
Test Newton's theorem that the gravitational attraction outside
a sphere is the same as if all its mass were concentrated at its center.
"""

from driver_spheregravity import run_spheregravity
from plot_spheregravity import plot_spheregravity
from physics_spheregravity import OutputType


# User-adjustable settings
nDiv = 100
outputType: OutputType = "relative difference"

radius, accel = run_spheregravity(nDiv=nDiv, outputType=outputType)
plot_spheregravity(radius, accel, outputType=outputType)
