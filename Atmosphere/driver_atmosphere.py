"""
Atmosphere driver module

Driver performs numerical interpolation using finite steps
in altitude, hydrostatic equilibrium, ideal gas law, and temperature.
"""

from dataclasses import dataclass
from bisect import bisect_left
import math
from numbers import Real
from typing import List, Literal, Sequence

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
# Cap |ΔT|/min(T) across one Euler step inside the supplied profile so a
# steep linear ramp cannot be crossed in a single sample.  Slowly varying
# teaching profiles (the default Earth run) never hit this cap.
MAX_RELATIVE_TEMP_JUMP = 0.05


@dataclass
class AtmosphereParameters:
    planet_name: str
    g_accel: float          # surface gravity (m/s^2)
    mu: float               # mean molecular mass (atomic mass units, u)
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
    mu: float | None = None


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


@dataclass(frozen=True)
class CheckpointData:
    """Atmospheric quantities at one supplied temperature-profile checkpoint."""
    altitude: float
    pressure: float | None
    pressure_over_temperature: float | None
    temperature: float
    density: float | None = None


class AtmosphereModel:
    def __init__(self, params: AtmosphereParameters):
        """Validate ``params`` and keep a model-owned copy of them.

        The model copies the altitude and temperature lists (and the scalar
        values), so changing the caller's lists or parameter object after
        construction does not change what ``run()`` computes.  ``run()``
        validates the model's own copy again before integrating.
        """
        if not isinstance(params, AtmosphereParameters):
            raise ValueError("params must be an AtmosphereParameters object.")
        self.params = params
        self.temp_profile = TemperatureProfile(
            h=params.h_points,
            T=params.T_points,
        )
        self._validate_parameters()
        self.params = AtmosphereParameters(
            planet_name=params.planet_name,
            g_accel=params.g_accel,
            mu=params.mu,
            p0=params.p0,
            h_points=list(params.h_points),
            T_points=list(params.T_points),
            output_type=params.output_type,
        )
        self.temp_profile = TemperatureProfile(
            h=self.params.h_points,
            T=self.params.T_points,
        )

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

    def _temperature_limited_step(
        self,
        altitude: float,
        temperature: float,
        proposed: float,
        pressure: float,
    ) -> float:
        """Shrink ``proposed`` when temperature would jump too far in one Euler step."""
        if proposed <= 0.0 or temperature <= 0.0:
            return proposed
        target = altitude + proposed
        try:
            t_target = self.temp_profile.get_temp(target, pressure)
        except ValueError:
            return proposed
        t_floor = min(temperature, t_target)
        if t_floor <= 0.0 or not math.isfinite(t_target):
            return proposed
        relative = abs(t_target - temperature) / t_floor
        if relative <= MAX_RELATIVE_TEMP_JUMP:
            return proposed
        # A landing a few micrometres wide cannot change ln p by a useful
        # amount even at the coldest temperature; leave those micro-steps
        # alone so a dense sounding still fits in the point budget.
        local_scale = t_floor / (
            self.params.g_accel * self.params.mu * phys.ATOMIC_MASS_UNIT / phys.K_BOLTZMANN
        )
        if local_scale > 0.0 and proposed / local_scale < 1e-5:
            return proposed
        return proposed * MAX_RELATIVE_TEMP_JUMP / relative

    def run(self) -> AtmosphereResult:
        """
        Compute an atmosphere profile by finite steps in altitude:

        - Compute the scale height at the coldest supplied temperature and
          the initial step dh from it
        - Use while-loop to adjust dh if top not reached within array size
        - Use for-loop to step in altitude, stopping when pressure <= 0
        - At each step: hydrostatic equilibrium, getTemp, ideal gas law

        The model's own copy of the parameters is validated again first, so
        an edit made to ``model.params`` after construction is either
        honoured with valid values or rejected with a clear ``ValueError``.
        """
        if not isinstance(self.params, AtmosphereParameters):
            raise ValueError("params must be an AtmosphereParameters object.")
        self.temp_profile = TemperatureProfile(
            h=self.params.h_points,
            T=self.params.T_points,
        )
        self._validate_parameters()
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

        # Initial altitude step: Schutz's choice of 200 steps per scale
        # height, but measured at the coldest temperature the supplied
        # profile reaches at or above the reference level.  The scale height
        # is proportional to T, so a layer colder than the surface has a
        # smaller local scale height, and a step sized from T(0) alone would
        # be too coarse there.  For an isothermal profile the coldest
        # temperature is T0 and the step is the same as before.
        coldest = min(
            [T0] + [t for h_i, t in zip(self.params.h_points, self.params.T_points) if h_i > 0.0]
        )
        scale_ref = p0 / (g * ideal_gas_density(p0, mu, coldest))
        if not math.isfinite(scale_ref) or scale_ref <= 0.0:
            raise ValueError("The supplied values do not produce a finite positive scale height.")
        dh = scale_ref / STEPS_PER_SCALE_HEIGHT
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

        # Interior temperature breakpoints above the reference level.  The
        # integrator lands on each of them so a thin layer cannot sit unseen
        # between two Euler samples.  A further cap on |ΔT|/T across a step
        # keeps a steep ramp from being crossed in one sample.
        profile_nodes = [height for height in self.params.h_points if height > 0.0]

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
            node_index = 0

            for j in range(1, max_steps):
                step = dh
                while (
                    node_index < len(profile_nodes)
                    and profile_nodes[node_index] <= alt[j - 1]
                ):
                    node_index += 1
                if node_index < len(profile_nodes):
                    distance = profile_nodes[node_index] - alt[j - 1]
                    if 0.0 < distance <= dh:
                        step = distance
                if alt[j - 1] < self.temp_profile.h[-1]:
                    step = self._temperature_limited_step(
                        alt[j - 1], Temp[j - 1], step, p[j - 1]
                    )
                alt[j] = alt[j - 1] + step
                if not math.isfinite(alt[j]):
                    raise RuntimeError("Altitude overflowed during integration.")
                p[j] = hydrostatic_step(p[j - 1], rho[j - 1], g, step)

                # Stop when the Euler step reaches or crosses the model's
                # zero-pressure boundary.  The non-positive point is excluded.
                if p[j] <= 0.0:
                    # The first pass takes 200 steps per scale height at the
                    # coldest temperature, so it cannot reach zero pressure at
                    # or below the top supplied altitude.  After a restart has
                    # enlarged the step it can: the step has then become too
                    # coarse for the coldest layer, and reporting that
                    # crossing as the top of the atmosphere would be false.
                    if retry_count > 1 and alt[j - 1] <= self.temp_profile.h[-1]:
                        raise RuntimeError(
                            "The enlarged altitude step reached zero pressure "
                            "inside the supplied temperature profile, so the "
                            "result would be unreliable there. The profile is "
                            "too tall, or too cold in one layer, for the "
                            "point budget: shorten it, or raise the coldest "
                            "temperature."
                        )
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
            mu=mu,
        )


def _require_result_arrays(result: AtmosphereResult) -> None:
    """Raise ValueError unless ``result`` is a well-formed AtmosphereResult.

    The four arrays must be non-string sequences of finite real numbers of one
    common, nonzero length, and the altitudes must increase strictly.
    """
    if not isinstance(result, AtmosphereResult):
        raise ValueError("result must be an AtmosphereResult object.")
    arrays = (
        ("altitudes", result.altitudes),
        ("pressures", result.pressures),
        ("densities", result.densities),
        ("temperatures", result.temperatures),
    )
    for name, values in arrays:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"AtmosphereResult {name} must be a non-string sequence of numbers.")
    if not result.altitudes or any(len(values) != len(result.altitudes) for _, values in arrays):
        raise ValueError("AtmosphereResult arrays must be nonempty and co-indexed.")
    for name, values in arrays:
        if any(
            not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"AtmosphereResult {name} must contain only finite numbers.")
    if any(
        result.altitudes[index + 1] <= result.altitudes[index]
        for index in range(len(result.altitudes) - 1)
    ):
        raise ValueError("AtmosphereResult altitudes must be strictly increasing.")
    if not math.isfinite(result.altitudes[-1] - result.altitudes[0]):
        raise ValueError(
            "AtmosphereResult altitudes span a range too large for interpolation."
        )
    if any(pressure <= 0.0 for pressure in result.pressures):
        raise ValueError("AtmosphereResult pressures must be finite positive numbers.")
    if any(density <= 0.0 for density in result.densities):
        raise ValueError("AtmosphereResult densities must be finite positive numbers.")
    if any(temperature <= 0.0 for temperature in result.temperatures):
        raise ValueError(
            "AtmosphereResult temperatures must be finite numbers greater than zero kelvin."
        )
    if result.mu is not None and (
        not isinstance(result.mu, Real) or isinstance(result.mu, bool)
        or not math.isfinite(result.mu) or result.mu <= 0.0
    ):
        raise ValueError("AtmosphereResult mu must be None or a finite positive number.")


def extract_output(result: AtmosphereResult) -> CurveData:
    """
    x-values are altitude, y-values depend on outputType.
    """
    _require_result_arrays(result)
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


def extract_checkpoints(
    result: AtmosphereResult,
    h_points: List[float],
    T_points: List[float],
) -> List[CheckpointData]:
    """Interpolate pressure and density at every supplied temperature checkpoint.

    The checkpoint temperatures are the defining profile values themselves.
    Pressure is linearly interpolated between adjacent integration samples.
    For model results, density follows the ideal-gas law using that pressure,
    the model's molecular mass, and the supplied checkpoint temperature.
    Manually constructed results without molecular mass use interpolated
    stored density instead. The ratio p/T uses the same pressure and temperature.
    A checkpoint outside the stored positive-pressure domain remains in the
    returned table with unavailable pressure and density diagnostics.
    """
    TemperatureProfile(h=h_points, T=T_points).validate()
    _require_result_arrays(result)

    checkpoints = []
    first_altitude = result.altitudes[0]
    last_altitude = result.altitudes[-1]
    for altitude, temperature in zip(h_points, T_points):
        if altitude < first_altitude or altitude > last_altitude:
            checkpoints.append(
                CheckpointData(altitude, None, None, temperature, None)
            )
            continue

        upper = bisect_left(result.altitudes, altitude)
        if upper < len(result.altitudes) and result.altitudes[upper] == altitude:
            pressure = result.pressures[upper]
            stored_density = result.densities[upper]
        else:
            lower = upper - 1
            fraction = (
                (altitude - result.altitudes[lower])
                / (result.altitudes[upper] - result.altitudes[lower])
            )
            pressure = result.pressures[lower] + fraction * (
                result.pressures[upper] - result.pressures[lower]
            )
            stored_density = result.densities[lower] + fraction * (
                result.densities[upper] - result.densities[lower]
            )

        density = (
            ideal_gas_density(pressure, result.mu, temperature)
            if result.mu is not None else stored_density
        )

        checkpoints.append(
            CheckpointData(
                altitude=altitude,
                pressure=pressure,
                pressure_over_temperature=pressure / temperature,
                temperature=temperature,
                density=density,
            )
        )
    return checkpoints
