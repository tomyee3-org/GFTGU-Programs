"""
Find the orbits of small bodies around black holes and spherical stars.
Study relativistic effects, such as orbital precession and the existence
of an innermost stable circular orbit.

Edit the parameters below to explore different orbits around a solar-mass
black hole.  All quantities are in SI units (metres, seconds).

Physical context
----------------
The simulation integrates the equatorial geodesic of a test particle in
the Schwarzschild spacetime.  The GR correction adds a term  12K²/(c²r²)
to the Newtonian potential, where K = ½ u_init · x_init is the specific
angular momentum constant of the orbit.
"""

from driver_relativistic_orbit import (
    integrate_relativistic_orbit,
    RelativisticOrbitParams,
)
from plot_relativistic_orbit import plot_relativistic_orbit

# ---------------------------------------------------------------------------
# Parameters – edit these values to explore different orbits
# ---------------------------------------------------------------------------
params = RelativisticOrbitParams(
    x_init     = 1.5e4,    # initial x-position (m);  start on the x-axis
    u_init     = 1.2E8,    # initial y-velocity  (m/s); > 0 → counter-clockwise
    dt         = 2.0E-6,   # initial time-step   (s);  halved automatically near
                           #   periapsis or the horizon
    max_steps  = 6_000,    # safety cap on total integration steps
    max_orbits = 10,       # stop after this many complete orbits
    eps1       = 0.05,     # sets the accuracy of the time-step. If computed quantities
                           # change by a larger fraction than this in a time-step, the time-step
                           # will be cut in half, repeatedly if necessary.
    eps2       = 1.0e-4,   # corrector-loop convergence tolerance
)

# ---------------------------------------------------------------------------
# Run the integrator and plot the result
# ---------------------------------------------------------------------------
result = integrate_relativistic_orbit(params)

if result.fell_into_hole:
    print(f"Particle crossed the Schwarzschild horizon after "
          f"{result.final_step} steps.")
else:
    print(f"Integration complete: {result.n_orbits:.0f} orbit(s) "
          f"in {result.final_step} steps.")

plot_relativistic_orbit(result)
