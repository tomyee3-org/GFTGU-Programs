"""
Atmosphere driver module

Driver performs numerical interpolation using finite steps
in altitude, hydrostatic equilibrium, ideal gas law, and temperature.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import List, Literal

import physics_atmosphere as phys
from physics_atmosphere import TemperatureProfile, ideal_gas_density, hydrostatic_step


OutputType = Literal["Pressure", "Density", "Temperature"]

# Preserve the intended Euler resolution for ordinary profiles.  Schutz's
# original Java implementation used 1,000 array elements, which forced the
# default extended Earth profile to restart with a much coarser step.  Python
# can comfortably retain enough points to avoid that large accuracy loss.
STEPS_PER_SCALE_HEIGHT = 200
MAX_STEPS = 50_000
MAX_RETRIES = 25


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
    model_version: str = phys.MODEL_VERSION
    build_id: str = phys.BUILD_ID


@dataclass
class CurveData:
    """
    Plot-ready atmospheric data: x/y are the plotted series and the
    remaining fields provide units, labels, and the plot title.
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
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Raise ValueError with a clear message for invalid user inputs."""
        if not isinstance(self.params.planet_name, str) or not self.params.planet_name.strip():
            raise ValueError("planet_name must be a non-empty string.")

        for name, value in (("g_accel", self.params.g_accel),
                            ("mu", self.params.mu),
                            ("p0", self.params.p0)):
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.params.output_type not in ("Pressure", "Density", "Temperature"):
            raise ValueError(
                'output_type must be "Pressure", "Density", or "Temperature".'
            )
        self.temp_profile.validate()

    def run(self) -> AtmosphereResult:
        """
        Compute an atmosphere profile by finite steps in altitude:

        - Compute scale height and initial step dh
        - Use while-loop to adjust dh if top not reached within array size
        - Use for-loop to step in altitude, stopping when pressure <= 0
        - At each step: hydrostatic equilibrium, getTemp, ideal gas law
        """
        g = self.params.g_accel
        mu = self.params.mu
        p0 = self.params.p0

        # The integration reference level is altitude zero.  This agrees with
        # T_points[0] for the normal h_points[0] == 0 case and also handles a
        # profile whose first measurement lies above or below the reference.
        T0 = self.temp_profile.get_temp(0.0, p0)

        # Ideal gas law to get density at bottom
        rho0 = ideal_gas_density(p0, mu, T0)

        # Scale height: for an isothermal atmosphere, pressure falls by a factor e
        scale = p0 / (g * rho0)

        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("The supplied values do not produce a finite positive scale height.")

        # Initial altitude step, following Schutz's choice of 200 steps per
        # base scale height.
        dh = scale / STEPS_PER_SCALE_HEIGHT
        if not math.isfinite(dh) or dh <= 0.0:
            raise ValueError("The supplied values do not produce a usable altitude step.")

        # A larger modern point budget prevents the default 500 km Earth
        # profile from silently losing resolution through restart doubling.
        max_steps = MAX_STEPS
        alt = [0.0] * max_steps
        p = [0.0] * max_steps
        rho = [0.0] * max_steps
        Temp = [0.0] * max_steps

        alt[0] = 0.0
        p[0] = p0
        Temp[0] = T0
        rho[0] = rho0

        last_step = 0
        retry_count = 0
        max_retries = MAX_RETRIES

        # Outer while-loop: repeat with larger dh if we do not reach the
        # numerical upper boundary within max_steps.
        while last_step == 0:
            if retry_count >= max_retries:
                raise RuntimeError(
                    "Could not reach the numerical zero-pressure boundary "
                    "after repeated step-size increases."
                )
            retry_count += 1
            # Each retry is a fresh integration. The upper-atmosphere
            # extrapolation coefficient must therefore be recomputed.
            self.temp_profile.reached_top = False
            self.temp_profile.beta = 0.0

            for j in range(1, max_steps):
                alt[j] = alt[j - 1] + dh
                if not math.isfinite(alt[j]):
                    raise RuntimeError("Altitude overflowed during integration.")
                p[j] = hydrostatic_step(p[j - 1], rho[j - 1], g, dh)

                # Stop when the Euler step reaches or crosses the model's
                # zero-pressure boundary.  The non-positive point is excluded.
                if p[j] <= 0.0:
                    last_step = j
                    break

                Temp[j] = self.temp_profile.get_temp(alt[j], p[j])
                rho[j] = ideal_gas_density(p[j], mu, Temp[j])

            # If still zero, all steps were used without crossing zero pressure: increase dh.
            if last_step == 0:
                dh *= 2.0
                if not math.isfinite(dh):
                    raise RuntimeError("Altitude step overflowed during restart doubling.")

        # Prepare output arrays up to last_step (excluding the non-positive point)
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
            model_version=phys.MODEL_VERSION,
            build_id=phys.BUILD_ID,
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
    elif result.output_type == "Temperature":
        y = result.temperatures
        unit = "K"
    else:
        raise ValueError(
            'output_type must be "Pressure", "Density", or "Temperature".'
        )

    return CurveData(
        x=result.altitudes,
        y=y,
        y_unit=unit,
        x_label="altitude (m)",
        y_label=f"{result.output_type} ({unit})",
        title=f"{result.planet_name} atmosphere: {result.output_type}",
    )
