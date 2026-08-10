"""
Atmosphere driver module

Driver performs numerical interpolation using finite steps
in altitude, hydrostatic equilibrium, ideal gas law, and temperature.
"""

from dataclasses import dataclass
from typing import List, Literal

from physics_atmosphere import TemperatureProfile, ideal_gas_density, hydrostatic_step


OutputType = Literal["Pressure", "Density", "Temperature"]


@dataclass
class AtmosphereParameters:
    planet_name: str
    g_accel: float          # surface gravity (m/s^2)
    mu: float               # mean molecular weight (in proton masses)
    p0: float               # surface pressure (Pa)
    h_points: List[float]   # measured altitudes (m)
    T_points: List[float]   # measured temperatures (K)
    output_type: OutputType


@dataclass
class AtmosphereResult:
    altitudes: List[float]
    pressures: List[float]
    densities: List[float]
    temperatures: List[float]
    output_type: OutputType
    planet_name: str


@dataclass
class CurveData:
    """
    Structured, uniformly-typed replacement for the old
    Dict[str, List[float]] curve output: x/y are the plotted series,
    the rest are display strings for the plot.
    """
    x: List[float]
    y: List[float]
    y_unit: str
    x_label: str
    y_label: str
    title: str


class AtmosphereModel:
    def __init__(self, params: AtmosphereParameters):
        self.params = params
        self.temp_profile = TemperatureProfile(
            h=params.h_points,
            T=params.T_points,
        )

    def run(self) -> AtmosphereResult:
        """
        Follow the original Atmosphere logic:

        - Compute scale height and initial step dh
        - Use while-loop to adjust dh if top not reached within array size
        - Use for-loop to step in altitude, stopping when pressure < 0
        - At each step: hydrostatic equilibrium, getTemp, ideal gas law
        """
        g = self.params.g_accel
        mu = self.params.mu
        p0 = self.params.p0

        # Initial temperature at base from first measurement
        T0 = self.params.T_points[0]

        # Ideal gas law to get density at bottom
        rho0 = ideal_gas_density(p0, mu, T0)

        # Scale height: roughly distance over which pressure falls by factor ~2
        scale = p0 / (g * rho0)

        # Step size in altitude; Java uses scale / 200.
        dh = scale / 200.0

        # Arrays of fixed maximum length, as in Java (1000 elements)
        max_steps = 1000
        alt = [0.0] * max_steps
        p = [0.0] * max_steps
        rho = [0.0] * max_steps
        Temp = [0.0] * max_steps

        alt[0] = 0.0
        p[0] = p0
        Temp[0] = T0
        rho[0] = rho0

        last_step = 0
        self.temp_profile.reached_top = False

        # Outer while-loop: repeat with larger dh if we don't reach top
        while last_step == 0:
            for j in range(1, max_steps):
                alt[j] = alt[j - 1] + dh
                p[j] = hydrostatic_step(p[j - 1], rho[j - 1], g, dh)

                # Stop when pressure goes negative
                if p[j] < 0.0:
                    last_step = j
                    break

                Temp[j] = self.temp_profile.get_temp(alt[j], p[j])
                rho[j] = ideal_gas_density(p[j], mu, Temp[j])

            # If still zero, we used all steps without reaching top: increase dh
            dh *= 2.0

        # Prepare output arrays up to last_step (excluding the negative-pressure point)
        final_alt = alt[:last_step]
        final_p = p[:last_step]
        final_rho = rho[:last_step]
        final_T = Temp[:last_step]

        return AtmosphereResult(
            altitudes=final_alt,
            pressures=final_p,
            densities=final_rho,
            temperatures=final_T,
            output_type=self.params.output_type,
            planet_name=self.params.planet_name,
        )


def extract_output(result: AtmosphereResult) -> CurveData:
    """
    x-values are altitude, y-values depend on outputType.
    """
    if result.output_type == "Pressure":
        y = result.pressures
        unit = "Pa"
    elif result.output_type == "Density":
        y = result.densities
        unit = "kg/m^3"
    else:  # "Temperature"
        y = result.temperatures
        unit = "K"

    return CurveData(
        x=result.altitudes,
        y=y,
        y_unit=unit,
        x_label="altitude (m)",
        y_label=f"{result.output_type} ({unit})",
        title=f"{result.planet_name} atmosphere: {result.output_type}",
    )
