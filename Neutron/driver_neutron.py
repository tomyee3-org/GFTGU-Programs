"""
Driver module for Neutron star structure.

Implements:
- initial special step at r = dr
- loop until pressure becomes negative
- automatic doubling of dr if surface not reached within 2000 steps
"""

import math
from physics_neutron import eos_density, tov_pressure_step, mass_step, G

def compute_neutron_star(gamma, pC, K):
    """
    Returns structured output:
    {
        "radius": [...],
        "pressure": [...],
        "density": [...],
        "mass": [...],
        "last_step": N
    }
    """

    gammaRecip = 1.0 / gamma
    rhoC = (pC / K) ** gammaRecip
    scale = math.sqrt(pC / G) / rhoC
    dr = scale / 400.0

    # Allocate arrays
    NMAX = 2000
    radius = [0.0] * NMAX
    pressure = [0.0] * NMAX
    density = [0.0] * NMAX
    mass = [0.0] * NMAX

    # Initial values
    radius[0] = 0.0
    pressure[0] = pC
    density[0] = rhoC
    mass[0] = 0.0

    lastStep = 0

    # Outer loop: repeat with larger dr if needed
    while lastStep == 0:

        # First non-zero step (special approximation)
        radius[1] = dr
        pressure[1] = pC
        density[1] = rhoC
        mass[1] = 4.0 * math.pi * dr**3 * rhoC / 3.0

        # Main loop
        for j in range(2, NMAX):
            radius[j] = radius[j-1] + dr

            # Hydrostatic equilibrium (relativistic)
            pressure[j] = tov_pressure_step(
                pressure[j-1],
                density[j-1],
                mass[j-1],
                radius[j-1],
                dr
            )

            # Stop when pressure becomes negative
            if pressure[j] < 0.0:
                lastStep = j
                break

            # Mass update
            mass[j] = mass_step(
                mass[j-1],
                density[j-1],
                radius[j-1],
                dr
            )

            # EOS update
            density[j] = eos_density(pressure[j], K, gamma)

        # If surface not reached, double dr and repeat
        dr *= 2.0

    # Trim arrays to lastStep
    return {
        "radius": radius[:lastStep],
        "pressure": pressure[:lastStep],
        "density": density[:lastStep],
        "mass": mass[:lastStep],
        "last_step": lastStep
    }
