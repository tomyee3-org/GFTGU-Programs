"""
MercPert main module (RK45 version)

Simulate the motion of a small planet in a solar system consisting of a star
like the Sun and a large planet more massive than Jupiter and closer to the Sun.
The program illustrates the way that massive planets "clean out" regions of the
planetary system near themselves. Spectacular interactions between the small
planet and the massive one are easy to achieve with the suggested initial data.

Sets example parameters (which the user can edit) and runs the simulation
using the RK45 adaptive integrator via scipy.integrate.solve_ivp.
"""

from physics_mercpert import AU
from driver_mercpert import run_mercpert
from plot_mercpert import plot_orbits


def main():
    # ── Binary system parameters ────────────────────────────────────────────
    m_sun_solar       = 1.0        # Sun mass in solar masses
    m_planet_solar    = 0.1        # ~100 Jupiter masses
    binary_separation = 0.7 * AU   # centre-to-centre distance [m]

    # ── Mercury initial conditions ──────────────────────────────────────────
    x_init_merc  = 0.3 * AU        # initial x-position [m]
    y_init_merc  = 0.0             # initial y-position [m]
    vx_init_merc = 0.0             # initial x-velocity [m/s]
    vy_init_merc = 59220.0         # initial y-velocity [m/s]

    # ── Integration parameters ──────────────────────────────────────────────
    # t_max     : total simulation time in seconds
    #             1e8 s ≈ 3.2 years  (equivalent to original 50000 × 2000 s)
    # dt_output : spacing between recorded output points [s]
    #             The RK45 integrator chooses its own internal step sizes to
    #             satisfy rtol/atol; dt_output only controls how many points
    #             are stored for plotting.
    # rtol/atol : relative and absolute error tolerances for solve_ivp
    t_max     = 1.0e8              # [s]
    dt_output = 2000.0             # [s]
    rtol      = 1e-9
    atol      = 1e-9

    result = run_mercpert(
        m_sun_solar       = m_sun_solar,
        m_planet_solar    = m_planet_solar,
        binary_separation = binary_separation,
        x_init_merc       = x_init_merc,
        y_init_merc       = y_init_merc,
        vx_init_merc      = vx_init_merc,
        vy_init_merc      = vy_init_merc,
        t_max             = t_max,
        dt_output         = dt_output,
        rtol              = rtol,
        atol              = atol,
    )

    plot_orbits(result)


if __name__ == "__main__":
    main()
