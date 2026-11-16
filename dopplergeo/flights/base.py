"""Provider-independent flight data: TrackPoint rows, query windows, providers,
and per-aircraft ECEF trajectories usable by the Doppler forward model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicHermiteSpline

from ..core.geo import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    velocity_enu,
    velocity_enu_to_ecef,
)

if TYPE_CHECKING:
    from .cache import FlightCache

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    ("t", pa.float64()),            # UTC epoch seconds
    ("icao24", pa.string()),        # lower-case hex; readsb non-ICAO addresses keep their '~'
    ("callsign", pa.string()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_geo_m", pa.float32()),    # geometric (WGS84 ellipsoid) altitude
    ("alt_baro_m", pa.float32()),   # pressure altitude
    ("gs_mps", pa.float32()),
    ("track_deg", pa.float32()),
    ("vrate_mps", pa.float32()),
    ("on_ground", pa.bool_()),
])
COLUMNS = SCHEMA.names


@dataclass(frozen=True)
class TrackPoint:
    t: float
    icao24: str
    callsign: str | None
    lat: float
    lon: float
    alt_geo_m: float | None
    alt_baro_m: float | None
    gs_mps: float | None
    track_deg: float | None
    vrate_mps: float | None
    on_ground: bool = False


def points_to_table(points: Iterable[TrackPoint]) -> pa.Table:
    cols: dict[str, list] = {c: [] for c in COLUMNS}
    for p in points:
        for c in COLUMNS:
            cols[c].append(getattr(p, c))
    return pa.Table.from_pydict(cols, schema=SCHEMA)


def table_to_points(table: pa.Table) -> list[TrackPoint]:
    return [TrackPoint(**row) for row in table.select(COLUMNS).to_pylist()]


def empty_table() -> pa.Table:
    return SCHEMA.empty_table()


# ---------------------------------------------------------------------------
# Query geometry
# ---------------------------------------------------------------------------

KM_PER_DEG_LAT = 111.32


@dataclass(frozen=True)
class BBox:
    """Latitude/longitude box in degrees. Boxes crossing the antimeridian are not supported."""

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90 <= self.lat_min <= self.lat_max <= 90 and -180 <= self.lon_min <= self.lon_max <= 180):
            raise ValueError(f"invalid bbox {self}")

    @classmethod
    def world(cls) -> BBox:
        return cls(-90.0, -180.0, 90.0, 180.0)

    @classmethod
    def around(cls, lat: float, lon: float, radius_km: float) -> BBox:
        return cls(lat, lon, lat, lon).expanded(radius_km)

    @classmethod
    def parse(cls, text: str) -> BBox:
        """``lat_min,lon_min,lat_max,lon_max``."""
        vals = [float(x) for x in text.split(",")]
        if len(vals) != 4:
            raise ValueError(f"expected lat_min,lon_min,lat_max,lon_max, got {text!r}")
        return cls(*vals)

    def expanded(self, km: float) -> BBox:
        dlat = km / KM_PER_DEG_LAT
        worst_lat = min(89.0, max(abs(self.lat_min), abs(self.lat_max)) + dlat)
        dlon = km / (KM_PER_DEG_LAT * math.cos(math.radians(worst_lat)))
        return BBox(max(-90.0, self.lat_min - dlat), max(-180.0, self.lon_min - dlon),
                    min(90.0, self.lat_max + dlat), min(180.0, self.lon_max + dlon))

    @property
    def center(self) -> tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2

    @property
    def area_sq_deg(self) -> float:
        return (self.lat_max - self.lat_min) * (self.lon_max - self.lon_min)

    def covers(self, other: BBox) -> bool:
        return (self.lat_min <= other.lat_min and self.lon_min <= other.lon_min
                and self.lat_max >= other.lat_max and self.lon_max >= other.lon_max)

    def contains(self, lat: ArrayLike, lon: ArrayLike) -> NDArray[np.bool_]:
        lat, lon = np.asarray(lat), np.asarray(lon)
        return (lat >= self.lat_min) & (lat <= self.lat_max) & (lon >= self.lon_min) & (lon <= self.lon_max)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.lat_min, self.lon_min, self.lat_max, self.lon_max


@dataclass(frozen=True)
class FlightQuery:
    t0: float
    t1: float
    bbox: BBox

    def __post_init__(self) -> None:
        if self.t1 <= self.t0:
            raise ValueError("t1 must be after t0")

    def utc_dates(self) -> list[datetime]:
        """Every UTC calendar day the window touches (midnight datetimes)."""
        d = datetime.fromtimestamp(self.t0, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        out = []
        while d.timestamp() < self.t1:
            out.append(d)
            d = datetime.fromtimestamp(d.timestamp() + 86400, tz=timezone.utc)
        return out


def query_window(start_utc: float, duration_s: float, region: BBox, clock_bound_s: float = 0.0,
                 pad_s: float = 120.0, expand_km: float = 20.0) -> FlightQuery:
    """Spec access strategy step 1: recording span ± 2 min ± clock bound, region + 20 km."""
    margin = pad_s + abs(clock_bound_s)
    return FlightQuery(start_utc - margin, start_utc + duration_s + margin, region.expanded(expand_km))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """A provider failed or cannot serve a request; the service moves on to the next one."""


class ProviderUnavailable(ProviderError):
    """The provider cannot serve this window at all (too old, no credentials, ...)."""


@dataclass(frozen=True)
class Coverage:
    """A (time, area) region the cache can now answer completely for one provider."""

    provider: str
    t0: float
    t1: float
    bbox: BBox


class FlightProvider(ABC):
    name: str = ""
    attribution: str = ""

    def can_serve(self, query: FlightQuery, now: float) -> str | None:
        """None if the provider can try this query, else the reason it cannot."""
        return None

    @abstractmethod
    def fetch(self, query: FlightQuery, cache: FlightCache) -> list[Coverage]:
        """Fetch data for ``query`` into ``cache`` and return the coverage it added."""


# ---------------------------------------------------------------------------
# Per-aircraft trajectories
# ---------------------------------------------------------------------------


def _nan(a: ArrayLike) -> FloatArray:
    a = np.asarray(a)
    return _none_to_nan(a) if a.dtype == object else a.astype(float)


@dataclass
class FlightTrack:
    """One aircraft leg as a smooth ECEF trajectory.

    Positions are interpolated with a cubic Hermite spline in ECEF. Node
    derivatives are the ADS-B ground speed / track / vertical rate where
    reported, else finite differences of position. ``position``/``velocity``
    match the ``TrajectoryFn`` signature used by ``core.doppler``.
    """

    icao24: str
    callsign: str | None
    t: FloatArray
    lat: FloatArray
    lon: FloatArray
    alt_m: FloatArray
    alt_source: str  # "geo", "baro+offset" or "baro"
    ecef: FloatArray = field(repr=False)
    vel_ecef: FloatArray = field(repr=False)
    gs_mps: FloatArray = field(repr=False)
    max_gap_s: float = 30.0
    _pos: CubicHermiteSpline = field(init=False, repr=False)
    _vel: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pos = CubicHermiteSpline(self.t, self.ecef, self.vel_ecef, axis=0, extrapolate=False)
        self._vel = self._pos.derivative()

    # -- construction -------------------------------------------------------

    @classmethod
    def from_arrays(cls, icao24: str, callsign: str | None, t: ArrayLike, lat: ArrayLike, lon: ArrayLike,
                    alt_geo_m: ArrayLike, alt_baro_m: ArrayLike, gs_mps: ArrayLike, track_deg: ArrayLike,
                    vrate_mps: ArrayLike, max_gap_s: float = 30.0) -> FlightTrack | None:
        t, lat, lon = _nan(t), _nan(lat), _nan(lon)
        geo, baro = _nan(alt_geo_m), _nan(alt_baro_m)
        gs, trk, vr = _nan(gs_mps), _nan(track_deg), _nan(vrate_mps)

        # Altitude: geometric if available; else baro shifted by this aircraft's
        # median geo-baro offset; else raw baro (flagged).
        both = np.isfinite(geo) & np.isfinite(baro)
        if np.isfinite(geo).all():
            alt, src = geo, "geo"
        elif both.any():
            off = float(np.median(geo[both] - baro[both]))
            alt, src = np.where(np.isfinite(geo), geo, baro + off), "baro+offset"
        else:
            alt, src = np.where(np.isfinite(geo), geo, baro), "baro"

        ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
        t, lat, lon, alt, gs, trk, vr = (a[ok] for a in (t, lat, lon, alt, gs, trk, vr))
        order = np.argsort(t, kind="stable")
        t, lat, lon, alt, gs, trk, vr = (a[order] for a in (t, lat, lon, alt, gs, trk, vr))
        if t.size < 2:
            return None
        # Keep the last report for duplicated timestamps.
        keep = np.append(np.diff(t) > 1e-3, True)
        t, lat, lon, alt, gs, trk, vr = (a[keep] for a in (t, lat, lon, alt, gs, trk, vr))
        if t.size < 2:
            return None

        ecef = geodetic_to_ecef(lat, lon, alt)
        fd = np.gradient(ecef, t, axis=0)
        # Vertical rate: fall back to the altitude derivative when not reported.
        vr = np.where(np.isfinite(vr), vr, np.gradient(alt, t))
        reported = velocity_enu_to_ecef(velocity_enu(gs, trk, vr), lat, lon)
        has_v = np.isfinite(gs) & np.isfinite(trk)
        vel = np.where(has_v[:, None], reported, fd)
        return cls(icao24, callsign, t, lat, lon, alt, src, ecef, vel, gs, max_gap_s)

    # -- queries ------------------------------------------------------------

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def position(self, t: ArrayLike) -> FloatArray:
        """ECEF position (n, 3); NaN outside the track span."""
        return np.asarray(self._pos(np.asarray(t, dtype=float)), dtype=float)

    def velocity(self, t: ArrayLike) -> FloatArray:
        """ECEF velocity (n, 3); NaN outside the track span."""
        return np.asarray(self._vel(np.asarray(t, dtype=float)), dtype=float)

    def reliable(self, t: ArrayLike) -> NDArray[np.bool_]:
        """True where t lies inside the span and the bracketing reports are ≤ max_gap_s apart."""
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, self.t.size - 2)
        inside = (t >= self.t[0]) & (t <= self.t[-1])
        return inside & (self.t[i + 1] - self.t[i] <= self.max_gap_s)

    def geodetic(self, t: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        return ecef_to_geodetic(self.position(t))

    def resample(self, step_s: float = 1.0) -> dict[str, FloatArray]:
        """Spec step 5: the trajectory on a regular grid with a reliability mask."""
        grid = np.arange(self.t[0], self.t[-1] + 1e-9, step_s)
        return {"t": grid, "ecef": self.position(grid), "vel": self.velocity(grid),
                "reliable": self.reliable(grid)}

    def __len__(self) -> int:
        return int(self.t.size)


def tracks_from_table(table: pa.Table, max_gap_s: float = 30.0, split_gap_s: float = 600.0,
                      include_ground: bool = False) -> list[FlightTrack]:
    """Group rows by aircraft, split legs at gaps > ``split_gap_s``, build trajectories."""
    if table.num_rows == 0:
        return []
    cols = {c: table.column(c).to_numpy(zero_copy_only=False) for c in COLUMNS}
    if not include_ground:
        keep = ~cols["on_ground"].astype(bool)
        cols = {c: v[keep] for c, v in cols.items()}
    icao = cols["icao24"].astype(str)
    order = np.lexsort((cols["t"].astype(float), icao))
    cols = {c: v[order] for c, v in cols.items()}
    icao = icao[order]
    t = cols["t"].astype(float)

    starts = np.flatnonzero(np.r_[True, (icao[1:] != icao[:-1]) | (np.diff(t) > split_gap_s)])
    ends = np.r_[starts[1:], icao.size]
    out: list[FlightTrack] = []
    for a, b in zip(starts, ends):
        calls = [c for c in cols["callsign"][a:b] if c]
        callsign = max(set(calls), key=calls.count) if calls else None
        tr = FlightTrack.from_arrays(
            str(icao[a]), callsign, t[a:b], cols["lat"][a:b], cols["lon"][a:b],
            _none_to_nan(cols["alt_geo_m"][a:b]), _none_to_nan(cols["alt_baro_m"][a:b]),
            _none_to_nan(cols["gs_mps"][a:b]), _none_to_nan(cols["track_deg"][a:b]),
            _none_to_nan(cols["vrate_mps"][a:b]), max_gap_s=max_gap_s)
        if tr is not None:
            out.append(tr)
    return out


def _none_to_nan(a: np.ndarray) -> FloatArray:
    if a.dtype == object:
        return np.array([np.nan if x is None else x for x in a], dtype=float)
    return a.astype(float)
