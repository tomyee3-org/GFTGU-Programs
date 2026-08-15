"""
Main entry point for Atmosphere

Compute the structure of the atmosphere of the Earth or of other
planets and moons in the Solar System. Find the way the density
and pressure depend on altitude.

User sets parameters here; they can overwrite the example values.
"""

from driver_atmosphere import AtmosphereParameters, AtmosphereModel, OutputType, extract_output
from plot_atmosphere import plot_atmosphere


def main():
    # Example: Earth's atmosphere, rough values
    planet_name = "Earth"
    g_accel = 9.81          # m/s^2
    mu = 28.97              # mean molecular weight ~ air (proton masses)
    p0 = 1.013e5            # surface pressure ~ 1 atm (Pa)

    # Simple temperature profile: (altitude, temperature) pairs
    # You can replace these with more detailed data (e.g., from Table 7.1).
    # Hybrid Earth temperature profile.
    #
    # 0-86 km:
    #   U.S. Standard Atmosphere 1976 layer-boundary temperatures.
    #   The standard defines these boundaries in geopotential height; the
    #   corresponding geometric altitudes are used here.
    #
    # 100-500 km:
    #   Rounded neutral-temperature values read from an NRLMSIS 2.0 CCMC
    #   Instant Run for 2024-10-15 12:00 UTC, 55 deg N, 45 deg E. CCMC was
    #   instructed to use observational F10.7 and Ap values for that date.
    #   These upper-atmosphere values are illustrative, not universal: the
    #   thermosphere varies strongly with solar and geomagnetic conditions,
    #   latitude, season, and time of day.
    h_points = [
        0.0,
        11_019.0,
        20_063.0,
        32_162.0,
        47_350.0,
        51_412.0,
        71_802.0,
        86_000.0,
        100_000.0,
        150_000.0,
        200_000.0,
        250_000.0,
        300_000.0,
        400_000.0,
        500_000.0,
    ]
    T_points = [
        288.15,
        216.65,
        216.65,
        228.65,
        270.65,
        270.65,
        214.65,
        186.946,
        190.0,
        800.0,
        1080.0,
        1190.0,
        1225.0,
        1240.0,
        1240.0,
    ]
    # Choose what to output: "Pressure", "Density", or "Temperature"
    output_type: OutputType = "Pressure"

    params = AtmosphereParameters(
        planet_name=planet_name,
        g_accel=g_accel,
        mu=mu,
        p0=p0,
        h_points=h_points,
        T_points=T_points,
        output_type=output_type,
    )

    model = AtmosphereModel(params)
    result = model.run()
    curve_data = extract_output(result)
    plot_atmosphere(curve_data)


if __name__ == "__main__":
    main()
