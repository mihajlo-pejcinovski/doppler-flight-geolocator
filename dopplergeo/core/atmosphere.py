"""Speed of sound in dry air, ISA temperature profile, and the air the sound
travels through (temperature + uniform wind) for the solver's forward model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA_AIR = 1.4
R_AIR = 287.05  # J/(kg K), specific gas constant of dry air
ISA_LAPSE_RATE = 0.0065  # K/m, troposphere (< 11 km)
_A = float(np.sqrt(GAMMA_AIR * R_AIR))  # c = _A * sqrt(T[K])


def speed_of_sound(temp_c: float | np.ndarray) -> float | np.ndarray:
    """Speed of sound in dry air (m/s) at temperature ``temp_c`` (deg C)."""
    return np.sqrt(GAMMA_AIR * R_AIR * (np.asarray(temp_c, dtype=float) + 273.15))


def temperature_for_speed(c: float) -> float:
    """Inverse of ``speed_of_sound``: deg C."""
    return c * c / (GAMMA_AIR * R_AIR) - 273.15


SPEED_OF_SOUND_15C: float = float(speed_of_sound(15.0))  # ~340.29 m/s


def isa_temperature(altitude_m: float | np.ndarray, surface_temp_c: float = 15.0) -> np.ndarray:
    """Temperature (deg C) at ``altitude_m`` using the ISA tropospheric lapse rate,
    anchored at ``surface_temp_c`` at altitude 0."""
    alt = np.clip(np.asarray(altitude_m, dtype=float), None, 11_000.0)
    return surface_temp_c - ISA_LAPSE_RATE * alt


def path_mean_speed_of_sound(
    h_low_m: float, h_high_m: float, surface_temp_c: float = 15.0, n: int = 64
) -> float:
    """Effective speed of sound for a path between two altitudes.

    Travel time is what matters, so this is the harmonic mean of c(h) along the
    path: (h_high - h_low) / integral(dh / c(h)). Returns c(h_low) when the two
    altitudes are equal.
    """
    if abs(h_high_m - h_low_m) < 1e-6:
        return float(speed_of_sound(isa_temperature(h_low_m, surface_temp_c)))
    h = np.linspace(h_low_m, h_high_m, n)
    inv_c = 1.0 / speed_of_sound(isa_temperature(h, surface_temp_c))
    return float((h_high_m - h_low_m) / np.trapezoid(inv_c, h))


@dataclass(frozen=True)
class Atmosphere:
    """Air between the observer and the aircraft.

    ``temp_c`` is the air temperature at the observer; with ``lapse`` it falls at
    the ISA rate with height, and each ray uses the travel-time (harmonic) mean
    speed of sound between the observer's and the aircraft's height. ``wind_e``,
    ``wind_n`` (m/s) are the air's velocity (the direction it blows *towards*),
    taken as uniform along the path. ``source`` says where the values came from.
    """

    temp_c: float = 15.0
    wind_e: float = 0.0
    wind_n: float = 0.0
    lapse: bool = True
    source: str = "standard"

    @property
    def c_surface(self) -> float:
        return float(speed_of_sound(self.temp_c))

    @property
    def wind(self) -> np.ndarray:
        return np.array([self.wind_e, self.wind_n, 0.0])

    def c_path(self, dh: np.ndarray) -> np.ndarray:
        """Travel-time mean speed of sound over a straight ray climbing ``dh`` metres
        from the observer. With T(z) = T0 - L z and c = A sqrt(T):
        integral(dz / c) = 2 (sqrt(T0) - sqrt(T0 - L dh)) / (A L), so
        c_mean = A L dh / (2 (sqrt(T0) - sqrt(T0 - L dh)))."""
        t0 = self.temp_c + 273.15
        dh = np.asarray(dh, dtype=float)
        if not self.lapse:
            return np.full(dh.shape, _A * np.sqrt(t0))
        dh = np.clip(dh, -1000.0, 11_000.0)
        L = ISA_LAPSE_RATE
        s0, s1 = np.sqrt(t0), np.sqrt(t0 - L * dh)
        small = np.abs(dh) < 1.0
        mean = _A * L * dh / (2.0 * np.where(small, 1.0, s0 - s1))
        return np.where(small, _A * np.sqrt(t0 - 0.5 * L * dh), mean)

    def describe(self) -> str:
        spd = float(np.hypot(self.wind_e, self.wind_n))
        frm = (np.degrees(np.arctan2(-self.wind_e, -self.wind_n)) % 360.0) if spd > 0 else 0.0
        return (f"{self.temp_c:.1f} °C (c {self.c_surface:.1f} m/s{', ISA lapse' if self.lapse else ''}), "
                f"wind {spd:.1f} m/s from {frm:.0f}° ({self.source})")


def wind_from_met(speed_mps: float, from_deg: float) -> tuple[float, float]:
    """Meteorological wind (speed, direction it blows FROM, degrees) -> (east, north)
    components of the air velocity."""
    r = np.radians(from_deg)
    return float(-speed_mps * np.sin(r)), float(-speed_mps * np.cos(r))
