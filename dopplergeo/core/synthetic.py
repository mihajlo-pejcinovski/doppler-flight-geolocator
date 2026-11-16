"""Synthetic flyovers with a known answer, for tests and tuning.

Coordinates are local ENU metres (x east, y north, z up) with the ground at
z = 0. Headings are degrees clockwise from north.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .atmosphere import SPEED_OF_SOUND_15C
from .doppler import FloatArray, frequency_from_trajectory, retarded_time


@dataclass(frozen=True)
class StraightLineFlight:
    """Constant-velocity source: ``position(tau) = p0 + tau * vel``."""

    p0: tuple[float, float, float]
    vel: tuple[float, float, float]

    def position(self, tau: ArrayLike) -> FloatArray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        return np.asarray(self.p0) + tau[:, None] * np.asarray(self.vel)

    def velocity(self, tau: ArrayLike) -> FloatArray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        return np.broadcast_to(np.asarray(self.vel, dtype=float), (tau.size, 3)).copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @classmethod
    def from_cpa(
        cls,
        speed: float,
        cpa_distance: float,
        tau0: float,
        heading_deg: float = 90.0,
        altitude: float | None = None,
        observer: ArrayLike = (0.0, 0.0, 0.0),
        right_side: bool = True,
    ) -> StraightLineFlight:
        """Level flight whose closest approach to ``observer`` is ``cpa_distance``
        (slant, metres) at emission time ``tau0``.

        ``altitude`` is the height of the track above the observer (default: 70%
        of the slant distance). ``right_side`` puts the observer on the right of
        the track; flipping it gives the mirror geometry.
        """
        obs = np.asarray(observer, dtype=float)
        h = 0.7 * cpa_distance if altitude is None else float(altitude)
        if not 0.0 <= h <= cpa_distance:
            raise ValueError("altitude above the observer must be between 0 and cpa_distance")
        horiz = np.sqrt(cpa_distance**2 - h**2)
        hdg = np.deg2rad(heading_deg)
        direction = np.array([np.sin(hdg), np.cos(hdg), 0.0])
        right = np.array([np.cos(hdg), -np.sin(hdg), 0.0])  # direction rotated 90 deg clockwise
        # Observer on the right of the track means the track lies to the observer's left.
        offset = -right if right_side else right
        p_cpa = obs + horiz * offset + np.array([0.0, 0.0, h])
        vel = speed * direction
        p0 = p_cpa - tau0 * vel
        return cls(p0=tuple(p0), vel=tuple(vel))


def synthesize_track(
    flight: StraightLineFlight,
    observer: ArrayLike,
    f0: float,
    t: ArrayLike,
    c: float = SPEED_OF_SOUND_15C,
    k: ArrayLike = 1.0,
    sigma_f: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Measured-frequency track (Hz) at reception times ``t`` with Gaussian noise."""
    f, _ = frequency_from_trajectory(t, f0, flight.position, flight.velocity, observer, c, k)
    if sigma_f > 0:
        f = f + np.random.default_rng(rng).normal(0.0, sigma_f, size=f.shape)
    return f


def add_wind_noise(
    x: FloatArray,
    fs: float,
    snr_db: float = 0.0,
    corner_hz: float = 40.0,
    gust_period_s: float = 4.0,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Add gusty wind rumble: red noise (1/f^2 above ``corner_hz``) with a slow
    random amplitude envelope. ``snr_db`` is signal power over wind power."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(rng)
    n = x.size
    sos = butter(2, corner_hz, btype="lowpass", fs=fs, output="sos")
    wind = sosfilt(sos, rng.normal(size=n))
    # Envelope: low-passed noise, rectified, around 1.
    sos_env = butter(1, 1.0 / gust_period_s, btype="lowpass", fs=fs, output="sos")
    env = 1.0 + 2.0 * np.abs(sosfilt(sos_env, rng.normal(size=n)) * np.sqrt(fs * gust_period_s))
    wind *= env
    p_sig = float(np.mean(x * x))
    wind *= np.sqrt(p_sig / 10 ** (snr_db / 10) / float(np.mean(wind * wind)))
    return x + wind


def render_audio(
    flight: StraightLineFlight,
    observer: ArrayLike,
    tones: Sequence[tuple[float, float]],
    fs: float = 16_000.0,
    duration: float = 60.0,
    t_start: float = 0.0,
    c: float = SPEED_OF_SOUND_15C,
    snr_db: float | None = 20.0,
    ground_reflection: float = 0.0,
    r_ref: float = 1000.0,
    rng: np.random.Generator | int | None = None,
    wander_rel: float = 0.0,
    wander_s: float = 4.0,
) -> tuple[FloatArray, FloatArray]:
    """Render a mono recording of a flyover. Returns ``(t, x)``.

    Each tone ``(rest_frequency_hz, amplitude)`` is emitted with phase
    ``2 pi f tau``; the observer hears it with phase ``2 pi f tau(t)`` where
    ``tau(t)`` is the retarded time, so the Doppler shift is exact. Amplitude
    falls as ``r_ref / r`` (spherical spreading, no air absorption).

    ``ground_reflection`` (0..1) adds the image-source path from a rigid ground
    at z = 0 (Lloyd's mirror). The observer's z is its height above ground.
    ``snr_db`` sets white noise relative to the mean signal power; None = clean.
    ``wander_rel`` adds a slow random modulation of the rest frequency (engine
    RPM drift, as on real recordings): RMS relative deviation ``wander_rel``,
    correlation time ~``wander_s`` (sum of 8 sinusoids, seeded by ``rng``).
    It is common to all tones (one engine), so harmonics keep their ratio.
    """
    obs = np.asarray(observer, dtype=float)
    n = int(round(duration * fs))
    t = t_start + np.arange(n) / fs
    gen = np.random.default_rng(rng)
    if wander_rel > 0.0:
        om = 2.0 * np.pi * gen.uniform(0.2, 1.5, 8) / wander_s
        ph = gen.uniform(0.0, 2.0 * np.pi, 8)
        amp_w = wander_rel * np.sqrt(2.0 / 8)

        def warped(tau: FloatArray) -> FloatArray:  # integral of (1 + m(tau)) d tau
            return tau - amp_w * np.sum(np.cos(om[:, None] * tau[None, :] + ph[:, None]) / om[:, None], axis=0)
    else:
        def warped(tau: FloatArray) -> FloatArray:
            return tau

    def path(o: FloatArray) -> FloatArray:
        tau = retarded_time(t, flight.position, o, c, flight.velocity)
        gain = r_ref / np.linalg.norm(flight.position(tau) - o, axis=-1)
        w = warped(tau)
        out = np.zeros(n)
        for freq, amp in tones:
            out += amp * gain * np.sin(2.0 * np.pi * freq * w)
        return out

    x = path(obs)
    if ground_reflection:
        x += ground_reflection * path(obs * np.array([1.0, 1.0, -1.0]))
    if snr_db is not None:
        p_sig = float(np.mean(x * x))
        noise_std = np.sqrt(p_sig / 10 ** (snr_db / 10))
        x = x + gen.normal(0.0, noise_std, size=n)
    return t, x


@dataclass(frozen=True)
class TurningFlight:
    """Constant-speed, constant-turn-rate flight (``turn_rate_deg_s`` > 0 turns right),
    with an optional constant climb rate. ``position(0) == p0``, heading ``heading0_deg`` at 0."""

    p0: tuple[float, float, float]
    speed: float
    heading0_deg: float
    turn_rate_deg_s: float
    climb_mps: float = 0.0

    def _psi(self, tau: FloatArray) -> FloatArray:
        return np.deg2rad(self.heading0_deg + self.turn_rate_deg_s * tau)

    def position(self, tau: ArrayLike) -> FloatArray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        w = np.deg2rad(self.turn_rate_deg_s)
        psi0, psi = np.deg2rad(self.heading0_deg), self._psi(tau)
        if abs(w) < 1e-12:
            x, y = self.speed * tau * np.sin(psi0), self.speed * tau * np.cos(psi0)
        else:
            x = self.speed / w * (np.cos(psi0) - np.cos(psi))
            y = self.speed / w * (np.sin(psi) - np.sin(psi0))
        return np.asarray(self.p0) + np.stack([x, y, self.climb_mps * tau], axis=-1)

    def velocity(self, tau: ArrayLike) -> FloatArray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        psi = self._psi(tau)
        return np.stack([self.speed * np.sin(psi), self.speed * np.cos(psi),
                         np.full(tau.shape, self.climb_mps)], axis=-1)


def adsb_samples(position, velocity, t: ArrayLike, lat0: float, lon0: float) -> dict[str, FloatArray]:
    """What an ADS-B receiver would log for a trajectory given in the ENU frame at
    (lat0, lon0, 0): geodetic position, ground speed, true track and vertical rate
    in the aircraft's own local frame. Keys match ``FlightTrack.from_arrays``."""
    from .geo import enu_rotation, enu_to_geodetic

    t = np.asarray(t, dtype=float)
    lat, lon, h = enu_to_geodetic(position(t), lat0, lon0)
    v_ecef = velocity(t) @ enu_rotation(lat0, lon0)
    loc = np.array([enu_rotation(a, b) @ v for a, b, v in zip(lat, lon, v_ecef)])
    return {"t": t, "lat": lat, "lon": lon, "alt_geo_m": h, "alt_baro_m": h,
            "gs_mps": np.hypot(loc[:, 0], loc[:, 1]),
            "track_deg": np.degrees(np.arctan2(loc[:, 0], loc[:, 1])) % 360.0,
            "vrate_mps": loc[:, 2]}
