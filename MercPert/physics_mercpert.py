"""
MercPert physics module

Only physical constants are literal numbers here.
"""

from dataclasses import dataclass
import math

# Physical constants (SI units)
G = 6.6726e-11          # Newton's gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.98847e30      # Mass of the Sun (kg)
AU = 1.495978707e11     # Astronomical unit (m)


@dataclass
class BinarySystemParams:
    m_sun_solar: float       # mass of central body in solar masses
    m_planet_solar: float    # mass of massive planet in solar masses
    binary_separation: float # separation between Sun and planet (m)


@dataclass
class MercuryInitialConditions:
    x_init: float
    y_init: float
    vx_init: float
    vy_init: float


def k_gravity() -> float:
    """
    kGravity = G * M_sun.
    This allows binary masses to be given in solar masses.
    """
    return G * M_SUN


def compute_binary_radii(params: BinarySystemParams) -> tuple[float, float]:
    """
    Compute orbital radii of Sun and Planet in the circular binary.

    Both bodies orbit their common center of mass (barycenter) -- neither
    sits still. By Newton's third law the two bodies pull on each other
    equally, so the center of mass stays fixed at the origin while each
    body traces its own circle around it, with radius inversely
    proportional to its own mass:

        rSun = mPlanet / (mSun + mPlanet) * binarySeparation
        rPlanet = binarySeparation - rSun

    so that mSun*rSun = mPlanet*rPlanet (equal and opposite "moments"
    about the barycenter) and rSun + rPlanet = binarySeparation always.
    A much heavier Sun traces a correspondingly tiny circle -- with the
    default 10:1 mass ratio, the Sun's radius is only 1/10th the
    planet's -- which is why the Sun's trajectory can look like it's
    barely moving in the plot, while the massive planet's circle is
    large and obvious.
    """
    m_sun = params.m_sun_solar
    m_planet = params.m_planet_solar
    sep = params.binary_separation

    r_sun = m_planet / (m_sun + m_planet) * sep
    r_planet = sep - r_sun
    return r_sun, r_planet


def compute_binary_angular_velocity(params: BinarySystemParams) -> float:
    """
    omega = sqrt( kGravity * (mSun + mPlanet) / binarySeparation^3 )
    with mSun, mPlanet in solar masses.
    """
    kg = k_gravity()
    m_sun = params.m_sun_solar
    m_planet = params.m_planet_solar
    sep = params.binary_separation

    return math.sqrt(kg * (m_sun + m_planet) / (sep ** 3))


def binary_positions(t: float,
                     params: BinarySystemParams) -> tuple[tuple[float, float],
                                                          tuple[float, float]]:
    """
    Return positions (x, y) of Sun and Planet at time t.

    Both bodies are always on opposite sides of the barycenter (origin),
    each moving on its own circle of radius rSun or rPlanet (see
    compute_binary_radii) at the same angular rate omega, exactly 180
    degrees out of phase -- when one is at angle theta, the other is at
    theta + pi, so the two positions and the origin are always collinear.

        xSun = -rSun * cos(omega * t)
        ySun = -rSun * sin(omega * t)
        xPlanet = rPlanet * cos(omega * t)
        yPlanet = rPlanet * sin(omega * t)
    """
    r_sun, r_planet = compute_binary_radii(params)
    omega = compute_binary_angular_velocity(params)

    c1 = math.cos(omega * t)
    s1 = math.sin(omega * t)

    x_sun = -r_sun * c1
    y_sun = -r_sun * s1
    x_planet = r_planet * c1
    y_planet = r_planet * s1

    return (x_sun, y_sun), (x_planet, y_planet)


def mercury_acceleration(t: float,
                         x_merc: float,
                         y_merc: float,
                         params: BinarySystemParams) -> tuple[float, float]:
    """
    Compute acceleration of Mercury due to Sun and Planet.

        xMercSun = xMerc - xSun
        yMercSun = yMerc - ySun
        xMercPlanet = xMerc - xPlanet
        yMercPlanet = yMerc - yPlanet

        rMercSun = sqrt(xMercSun^2 + yMercSun^2)
        rMercSun3 = rMercSun^3
        rMercPlanet = sqrt(xMercPlanet^2 + yMercPlanet^2)
        rMercPlanet3 = rMercPlanet^3

        axMerc = -kGravity * ( mSun * xMercSun / rMercSun3
                               + mPlanet * xMercPlanet / rMercPlanet3 )
        ayMerc = -kGravity * ( mSun * yMercSun / rMercSun3
                               + mPlanet * yMercPlanet / rMercPlanet3 )
    """
    (x_sun, y_sun), (x_planet, y_planet) = binary_positions(t, params)

    x_merc_sun = x_merc - x_sun
    y_merc_sun = y_merc - y_sun
    x_merc_planet = x_merc - x_planet
    y_merc_planet = y_merc - y_planet

    r_merc_sun = math.sqrt(x_merc_sun ** 2 + y_merc_sun ** 2)
    r_merc_planet = math.sqrt(x_merc_planet ** 2 + y_merc_planet ** 2)

    # Avoid division by zero in pathological cases
    if r_merc_sun == 0.0:
        r_merc_sun = 1e-30
    if r_merc_planet == 0.0:
        r_merc_planet = 1e-30

    r_merc_sun3 = r_merc_sun ** 3
    r_merc_planet3 = r_merc_planet ** 3

    kg = k_gravity()
    m_sun = params.m_sun_solar
    m_planet = params.m_planet_solar

    ax_merc = -kg * (m_sun * x_merc_sun / r_merc_sun3 +
                     m_planet * x_merc_planet / r_merc_planet3)
    ay_merc = -kg * (m_sun * y_merc_sun / r_merc_sun3 +
                     m_planet * y_merc_planet / r_merc_planet3)

    return ax_merc, ay_merc
