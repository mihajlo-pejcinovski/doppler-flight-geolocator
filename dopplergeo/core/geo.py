"""WGS84 geodesy: geodetic <-> ECEF <-> local ENU, plus ENU velocity helpers.

Pure NumPy (closed-form forward transform, Bowring iteration for the inverse).
Sub-millimetre round trips for anything between the geoid and cruise altitude.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)

KNOT = 0.514444  # m/s
FOOT = 0.3048  # m
FPM = FOOT / 60.0  # ft/min -> m/s


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, h: ArrayLike = 0.0) -> FloatArray:
    """Latitude/longitude in degrees, ellipsoidal height in metres -> ECEF (..., 3) metres."""
    la = np.radians(np.asarray(lat, dtype=float))
    lo = np.radians(np.asarray(lon, dtype=float))
    h = np.asarray(h, dtype=float)
    sl, cl = np.sin(la), np.cos(la)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    x = (n + h) * cl * np.cos(lo)
    y = (n + h) * cl * np.sin(lo)
    z = (n * (1.0 - WGS84_E2) + h) * sl
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(xyz: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """ECEF (..., 3) -> (lat deg, lon deg, h m)."""
    p_ = np.asarray(xyz, dtype=float)
    x, y, z = p_[..., 0], p_[..., 1], p_[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring initial guess, then a few Newton-like refinements on latitude.
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
                     p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3)
    def height(lat: FloatArray) -> FloatArray:  # valid at the poles too
        sl = np.sin(lat)
        return p * np.cos(lat) + z * sl - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sl * sl)

    for _ in range(3):
        sl = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height(lat))))
    return np.degrees(lat), np.degrees(lon), height(lat)


def enu_rotation(lat0: float, lon0: float) -> FloatArray:
    """3x3 matrix whose rows are the E, N, U unit vectors (in ECEF) at (lat0, lon0)."""
    la, lo = np.radians(lat0), np.radians(lon0)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


def ecef_to_enu(xyz: ArrayLike, lat0: float, lon0: float, h0: float = 0.0) -> FloatArray:
    origin = geodetic_to_ecef(lat0, lon0, h0)
    return (np.asarray(xyz, dtype=float) - origin) @ enu_rotation(lat0, lon0).T


def enu_to_ecef(enu: ArrayLike, lat0: float, lon0: float, h0: float = 0.0) -> FloatArray:
    origin = geodetic_to_ecef(lat0, lon0, h0)
    return np.asarray(enu, dtype=float) @ enu_rotation(lat0, lon0) + origin


def geodetic_to_enu(lat: ArrayLike, lon: ArrayLike, h: ArrayLike, lat0: float, lon0: float,
                    h0: float = 0.0) -> FloatArray:
    return ecef_to_enu(geodetic_to_ecef(lat, lon, h), lat0, lon0, h0)


def enu_to_geodetic(enu: ArrayLike, lat0: float, lon0: float, h0: float = 0.0
                    ) -> tuple[FloatArray, FloatArray, FloatArray]:
    return ecef_to_geodetic(enu_to_ecef(enu, lat0, lon0, h0))


def velocity_enu(gs_mps: ArrayLike, track_deg: ArrayLike, vrate_mps: ArrayLike) -> FloatArray:
    """Ground speed / true track / vertical rate -> local (E, N, U) velocity (..., 3)."""
    tr = np.radians(np.asarray(track_deg, dtype=float))
    gs = np.asarray(gs_mps, dtype=float)
    return np.stack(np.broadcast_arrays(gs * np.sin(tr), gs * np.cos(tr),
                                        np.asarray(vrate_mps, dtype=float)), axis=-1)


def velocity_enu_to_ecef(v_enu: ArrayLike, lat: ArrayLike, lon: ArrayLike) -> FloatArray:
    """Rotate per-point local ENU velocities (..., 3) into ECEF, each at its own lat/lon."""
    la = np.radians(np.asarray(lat, dtype=float))
    lo = np.radians(np.asarray(lon, dtype=float))
    v = np.asarray(v_enu, dtype=float)
    e, n, u = v[..., 0], v[..., 1], v[..., 2]
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    x = -so * e - sl * co * n + cl * co * u
    y = co * e - sl * so * n + cl * so * u
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> FloatArray:
    """Great-circle distance on the mean sphere (R = 6371 km); good to ~0.5% for gating."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
