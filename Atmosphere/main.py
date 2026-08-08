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
    h_points = [
        0.0,        # Troposphere
        5_000.0,    # Troposphere
        10_000.0,   # Troposphere
        11_000.0,   # Tropopause (End of troposphere)
        20_000.0,   # Stratosphere
        30_000.0,   # Stratosphere
        47_000.0,   # Stratopause
        50_000.0,   # Mesosphere
        60_000.0,   # Mesosphere
        85_000.0,   # Mesopause
        100_000.0,  # Thermosphere
        200_000.0,  # Thermosphere
        500_000.0,  # Thermosphere
    ]
    T_points = [
        288.0,  # ~15°C at sea level
        281.5,
        275.0,
        217.0,
        227.0,
        237.0,
        271.0,
        258.0,
        248.0,
        263.0,
        273.0,
        373.0,
        423.0,
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
