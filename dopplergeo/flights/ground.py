"""Ground height from ADS-B: aircraft taxiing report GNSS (ellipsoidal) altitude.

Each airport in the data becomes a 'site' whose ground height is the median
geometric altitude of on-ground reports minus the GNSS antenna height (top of
the fuselage, ~4-6 m on airliners). Away from a site the value is only a guess
at the terrain, so the sigma grows with distance. Good on flat land near an
airport (checked at Schiphol: 45.7 m reported, 41.2 m after the antenna
correction, ~40 m expected); use the DEM in hilly terrain.

Also selects the observer height source for the solver (``observer_elevation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray

from ..core.elevation import PHONE_HEIGHT_M, ConstantElevation, ElevationModel, regional_dem
from ..core.geo import haversine_m
from .base import BBox

FloatArray = NDArray[np.float64]

ADSB_ANTENNA_HEIGHT_M = 4.5  # GNSS antenna above the tarmac, typical narrow/wide-body


@dataclass
class AdsbGroundElevation:
    site_lat: FloatArray
    site_lon: FloatArray
    site_ground: FloatArray       # ellipsoidal ground height per site
    site_reports: NDArray[np.int64]
    base_sigma_m: float = 8.0     # 25 ft quantisation, antenna-height spread, site size
    terrain_m_per_km: float = 1.0  # flat-land guess for how terrain departs from the site
    max_sigma_m: float = 300.0
    above_ground_m: float = PHONE_HEIGHT_M
    source: str = "adsb-ground"

    def nearest(self, lat: ArrayLike, lon: ArrayLike) -> tuple[NDArray[np.intp], FloatArray]:
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        d = haversine_m(lat[..., None], lon[..., None], self.site_lat, self.site_lon)
        i = d.argmin(-1)
        return i, np.take_along_axis(d, i[..., None], -1)[..., 0]

    def height(self, lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
        i, dist = self.nearest(lat, lon)
        h = self.site_ground[i] + self.above_ground_m
        sigma = np.minimum(np.hypot(self.base_sigma_m, self.terrain_m_per_km * dist / 1000.0), self.max_sigma_m)
        return h, sigma


def adsb_ground_elevation(table: pa.Table, cell_deg: float = 0.05, min_reports: int = 30,
                          antenna_height_m: float = ADSB_ANTENNA_HEIGHT_M) -> AdsbGroundElevation | None:
    """Sites from on-ground reports with a geometric altitude; None if there are none."""
    if table.num_rows == 0:
        return None
    gnd = table.column("on_ground").to_numpy(zero_copy_only=False).astype(bool)
    geo = table.column("alt_geo_m").to_numpy(zero_copy_only=False).astype(float)
    ok = gnd & np.isfinite(geo)
    if ok.sum() < min_reports:
        return None
    lat = table.column("lat").to_numpy()[ok]
    lon = table.column("lon").to_numpy()[ok]
    geo = geo[ok]
    key = np.floor(lat / cell_deg).astype(np.int64) * 100_000 + np.floor(lon / cell_deg).astype(np.int64)
    sites = []
    for k in np.unique(key):
        m = key == k
        if m.sum() < min_reports:
            continue
        h = geo[m]
        med = np.median(h)
        h = h[np.abs(h - med) < 50.0]  # drop junk (e.g. -518 m) before the final median
        if h.size >= min_reports:
            sites.append((lat[m].mean(), lon[m].mean(), float(np.median(h)) - antenna_height_m, h.size))
    if not sites:
        return None
    a = np.array(sites)
    return AdsbGroundElevation(a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.int64))


def observer_elevation(mode: str, region: BBox, table: pa.Table | None = None,
                       cache_dir: Path | str | None = None, constant_h: float = 0.0,
                       margin_km: float = 10.0) -> tuple[ElevationModel, list[str]]:
    """Pick the observer height source. ``mode``: auto | dem | adsb | constant.

    auto tries the DEM (accurate anywhere), then ADS-B airport ground, then the
    constant. Returns the model and notes on why sources were skipped.
    """
    notes: list[str] = []
    if mode in ("auto", "dem"):
        try:
            # Used by regional_dem only for a very large region (its grid search heights).
            fallback = (adsb_ground_elevation(table) if table is not None else None) \
                or ConstantElevation(constant_h, 1500.0, "unknown")
            return regional_dem(region.expanded(margin_km).as_tuple(), cache_dir=cache_dir,
                                fallback=fallback), notes
        except Exception as e:  # missing rasterio, offline, ...
            notes.append(f"DEM unavailable: {e}")
            if mode == "dem":
                raise
    if mode in ("auto", "adsb"):
        model = adsb_ground_elevation(table) if table is not None else None
        if model is not None:
            return model, notes
        notes.append("no on-ground ADS-B reports with geometric altitude in the data")
        if mode == "adsb":
            raise RuntimeError(notes[-1])
    return ConstantElevation(constant_h, 100.0, "constant"), notes
