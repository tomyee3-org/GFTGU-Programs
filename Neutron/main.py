"""
Build a model of a neutron star using the full structure equations of
general relativity. The user can choose the central pressure and some
aspects of the equation of state. The program will output graphs of the
density, pressure, or mass as functions of the radius.

Main entry point for Neutron star model.
User may edit gamma, pC, K, and output_type.
"""

from driver_neutron import compute_neutron_star
from plot_neutron import plot_neutron

if __name__ == "__main__":

    # Example values (user may overwrite). These default values for
    # the simplified model presented here yield results that are
    # reasonably in accordance with more realistic neutron star
    # models.

    gamma = 1.666667     # polytropic exponent (best fit to dense neutron
                          # matter, per Schutz's documentation)
    pC = 1.26e35         # central pressure (Pa)

    # K is the proportionality constant in p = K*rho^gamma. This value
    # implies a central density of ~6.6e18 kg/m^3 (~29x nuclear density),
    # and independently matches (to 4 significant figures) the
    # K ~ 5380.3 derivable from the physical equation of state of a
    # non-relativistic degenerate neutron gas,
    #   K = hbar^2/(15*pi^2*m_n) * (3*pi^2)^(5/3) / m_n^(5/3)
    # (hbar = reduced Planck constant, m_n = neutron mass).

    # With gamma and pC above this produces a ~7.1 km, ~1.1 solar-mass
    # star -- consistent with real neutron star models.

    K = 5.3802e3          # polytropic constant (SI)

    output_type = "Pressure"   # "Pressure", "Density", or "Mass"

    data = compute_neutron_star(gamma, pC, K)
    plot_neutron(data, output_type)
