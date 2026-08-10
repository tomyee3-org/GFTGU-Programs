"""
Atmosphere physics module
"""

from dataclasses import dataclass
from typing import List

# Physical constants (SI)
K_BOLTZMANN = 1.38e-23  # Boltzmann constant, J/K
M_PROTON = 1.67e-27     # Proton mass, kg


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

    def get_temp(self, altitude: float, pressure: float) -> float:
        """
        Interpolate temperature at given altitude. For altitudes above the
        highest measurement, use T = beta * p^power, with beta fixed so that
        T matches the last measured temperature at the altitude where the
        atmosphere first reaches the top.
        """
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
            # T_last = beta * p_last^power at the top of the measured region.
            t_last = self.T[-1]
            p_last = pressure
            if p_last > 0.0:
                self.beta = t_last / (p_last ** self.power)
            else:
                # Avoid division by zero; keep temperature constant if pressure ~ 0
                self.beta = t_last
            self.reached_top = True

        # Use power-law relation T = beta * p^power
        if pressure > 0.0:
            return self.beta * (pressure ** self.power)
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
    q = M_PROTON * mu / K_BOLTZMANN
    return pressure * q / temperature


def hydrostatic_step(pressure_prev: float, rho_prev: float, g_accel: float, dh: float) -> float:
    """
    One step of the hydrostatic equilibrium equation:

        p[j] = p[j-1] - gAccel * rho[j-1] * dh
    """
    return pressure_prev - g_accel * rho_prev * dh
