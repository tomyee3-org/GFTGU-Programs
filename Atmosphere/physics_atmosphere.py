"""
Atmosphere physics module
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import List

MODEL_VERSION = "1.1.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_atmosphere.py",
    "driver_atmosphere.py",
    "main.py",
    "plot_atmosphere.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same declared
    version.  The hash is independent of LF versus CRLF line endings and
    frames each file with its name and length so file-boundary changes cannot
    collide with an unchanged concatenated byte stream.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located or decoded, as can happen in some
    frozen or zipped distributions.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# Physical constants (SI)
K_BOLTZMANN = 1.38e-23  # Boltzmann constant, J/K
M_PROTON = 1.67e-27     # Proton mass, kg


def _is_finite_number(value) -> bool:
    """Return True for finite real numbers, excluding booleans."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class TemperatureProfile:
    """
    Holds measured (altitude, temperature) pairs and provides interpolation.

    Altitudes h[i] in meters, temperatures T[i] in kelvin.
    """
    h: List[float]
    T: List[float]
    power: float = 0.5  # exponent in T = beta * p^power for upper atmosphere
    beta: float = 0.0   # will be set when top is reached
    reached_top: bool = False

    def validate(self) -> None:
        """Validate the supplied temperature profile."""
        if len(self.h) != len(self.T):
            raise ValueError("h_points and T_points must contain the same number of values.")
        if len(self.h) < 2:
            raise ValueError("At least two altitude-temperature points are required.")
        if any(not _is_finite_number(h) for h in self.h):
            raise ValueError("h_points must contain only finite numbers.")
        if any(not _is_finite_number(t) or t <= 0.0 for t in self.T):
            raise ValueError("T_points must contain only finite temperatures greater than zero kelvin.")
        if any(self.h[i + 1] <= self.h[i] for i in range(len(self.h) - 1)):
            raise ValueError("h_points must be in strictly increasing order.")
        if not _is_finite_number(self.power) or self.power <= 0.0:
            raise ValueError("power must be a finite positive number.")

    def get_temp(self, altitude: float, pressure: float) -> float:
        """
        Interpolate temperature at given altitude. For altitudes above the
        highest measurement, use T = beta * p^power, with beta fixed so that
        T is continuous at the first integration point above the highest
        supplied temperature measurement.
        """
        if not _is_finite_number(altitude):
            raise ValueError("altitude must be a finite number.")
        if not _is_finite_number(pressure) or pressure < 0.0:
            raise ValueError("pressure must be a finite non-negative number.")

        # If we are still below or within measured range, do linear interpolation
        if altitude <= self.h[-1]:
            # Find bracketing indices
            for i in range(len(self.h) - 1):
                h_low = self.h[i]
                h_high = self.h[i + 1]
                if h_low <= altitude <= h_high:
                    t_low = self.T[i]
                    t_high = self.T[i + 1]
                    # Linear interpolation in altitude
                    frac = (altitude - h_low) / (h_high - h_low)
                    return t_low + frac * (t_high - t_low)
            # If altitude is below first measurement, just use first temperature
            if altitude < self.h[0]:
                return self.T[0]
            # If altitude is exactly at last measurement
            return self.T[-1]

        # Above highest measurement: upper atmosphere model
        if not self.reached_top:
            # First time we go above the measured region: fix beta so that
            # the extrapolated temperature at this first computed point equals T_last.
            t_last = self.T[-1]
            p_last = pressure
            if p_last > 0.0:
                try:
                    pressure_factor = p_last ** self.power
                except OverflowError as exc:
                    raise ValueError(
                        "The upper-atmosphere pressure law overflowed."
                    ) from exc
                if not math.isfinite(pressure_factor) or pressure_factor <= 0.0:
                    raise ValueError(
                        "The upper-atmosphere pressure law is outside the usable numerical range."
                    )
                self.beta = t_last / pressure_factor
                if not math.isfinite(self.beta) or self.beta <= 0.0:
                    raise ValueError(
                        "The upper-atmosphere coefficient is outside the usable numerical range."
                    )
            else:
                # Avoid division by zero; keep temperature constant if pressure ~ 0
                self.beta = t_last
            self.reached_top = True

        # Use power-law relation T = beta * p^power
        if pressure > 0.0:
            try:
                temperature = self.beta * (pressure ** self.power)
            except OverflowError as exc:
                raise ValueError(
                    "The upper-atmosphere temperature overflowed."
                ) from exc
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    "The upper-atmosphere temperature is outside the usable numerical range."
                )
            return temperature
        else:
            # If pressure has gone to zero or negative, keep last meaningful T
            return self.T[-1]


def ideal_gas_density(pressure: float, mu: float, temperature: float) -> float:
    """
    Ideal gas law in the form used by Atmosphere:

        rho = p * q / T,  where q = mp * mu / k

    pressure: p (Pa)
    mu: mean molecular weight (in units of proton mass)
    temperature: T (K)
    """
    if not _is_finite_number(pressure) or pressure < 0.0:
        raise ValueError("pressure must be a finite non-negative number.")
    if not _is_finite_number(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive number.")
    if not _is_finite_number(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite number greater than zero kelvin.")

    q = M_PROTON * mu / K_BOLTZMANN
    density = pressure * q / temperature
    if not math.isfinite(density):
        raise ValueError("The supplied values produce a non-finite density.")
    if pressure > 0.0 and density == 0.0:
        raise ValueError("The supplied values produce a density that underflows to zero.")
    return density


def hydrostatic_step(pressure_prev: float, rho_prev: float, g_accel: float, dh: float) -> float:
    """
    One step of the hydrostatic equilibrium equation:

        p[j] = p[j-1] - gAccel * rho[j-1] * dh
    """
    if not _is_finite_number(pressure_prev) or pressure_prev < 0.0:
        raise ValueError("pressure_prev must be a finite non-negative number.")
    if not _is_finite_number(rho_prev) or rho_prev < 0.0:
        raise ValueError("rho_prev must be a finite non-negative number.")
    if not _is_finite_number(g_accel) or g_accel <= 0.0:
        raise ValueError("g_accel must be a finite positive number.")
    if not _is_finite_number(dh) or dh <= 0.0:
        raise ValueError("dh must be a finite positive number.")

    pressure = pressure_prev - g_accel * rho_prev * dh
    if not math.isfinite(pressure):
        raise ValueError("The hydrostatic step produced a non-finite pressure.")
    return pressure
