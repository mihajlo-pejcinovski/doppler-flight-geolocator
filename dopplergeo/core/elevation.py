"""Observer height sources.

Every model returns the *observer's* height above the WGS84 ellipsoid (the
frame ADS-B geometric altitudes use) plus a 1-sigma uncertainty, for arrays of
lat/lon. The solver uses the value per grid cell and, when it fits a height
correction, the sigma as a prior.

* ``ConstantElevation``: a fixed value (user input, or the 0 m fallback).
* ``GridElevation``: a regular lat/lon grid, bilinear; ``copernicus_dem``
  builds one from Copernicus GLO-30 (orthometric, EGM2008) + the EGM2008
  geoid, read remotely as Cloud-Optimised GeoTIFFs (needs ``rasterio``).
* ``PatchedDem``: for regions too large for one 100 m grid (``regional_dem``):
  a coarse grid, capped at ``MAX_DEM_CELLS``, for whole-region queries (the
  solver's grid search) plus 100 m patches loaded on demand where the solver
  works locally (refinement, ellipse probes). Memory stays bounded whatever
  the region size.
* ``dopplergeo.flights.ground.AdsbGroundElevation``: airport surface height
  from aircraft taxiing (no downloads; only good near airports / flat land).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

PHONE_HEIGHT_M = 1.5  # handheld recorder above the ground

GLO30_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_DEM/"
             "Copernicus_DSM_COG_10_{tile}_DEM.tif")
EGM2008_URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
ATTRIBUTION_GLO30 = ("Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
                     "2014-2018, provided under COPERNICUS by the European Union and ESA")
# GLO-30 is a surface model (rooftops, canopy) with < 4 m LE90 absolute error.
GLO30_SIGMA_M = 6.0
FINE_RES_M = 100.0
# One 100 m grid costs ~40 bytes per cell while being built (measured: a 2 x 2 degree
# box, 3 M cells, peaked at 240 MB). Without a cap a large drawn region (15-20 degrees)
# needed ~16 GB. 1.5 M cells keeps the build near 100 MB.
MAX_DEM_CELLS = 1_500_000
PATCH_DEG = 0.1          # fine patches: 0.1 x 0.1 degree, aligned, cached on disk
MAX_LOCAL_PATCHES = 4    # a query touching more patches than this is a regional one
MAX_PATCHES_IN_MEMORY = 64
# A coarse grid at this resolution over the 1.5 M cell cap is a ~350 km box, about
# 20 remote 1-degree tiles (~2.5 s each). Larger regions skip the regional grid.
MAX_COARSE_RES_M = 250.0


class ElevationModel(Protocol):
    source: str

    def height(self, lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Observer height above the WGS84 ellipsoid (m) and its 1-sigma (m)."""
        ...


@dataclass
class ConstantElevation:
    h: float = 0.0
    sigma: float = 100.0
    source: str = "constant"

    def height(self, lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
        shape = np.broadcast(np.asarray(lat), np.asarray(lon)).shape
        return np.full(shape, self.h, dtype=float), np.full(shape, self.sigma, dtype=float)


@dataclass
class GridElevation:
    """Ground height above the ellipsoid on a regular grid (ascending lat, lon);
    the observer is ``above_ground_m`` higher. Outside the grid: the grid median
    with a wide sigma."""

    lat: FloatArray
    lon: FloatArray
    ground: FloatArray            # (lat, lon) ellipsoidal ground height
    sigma: float = GLO30_SIGMA_M
    source: str = "grid"
    above_ground_m: float = PHONE_HEIGHT_M
    attribution: str = ""
    _interp: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from scipy.interpolate import RegularGridInterpolator

        g = np.asarray(self.ground, dtype=float)
        fill = float(np.nanmedian(g)) if np.isfinite(g).any() else 0.0
        self._fill = fill
        self._interp = RegularGridInterpolator((self.lat, self.lon), np.where(np.isfinite(g), g, fill),
                                               bounds_error=False, fill_value=np.nan)

    def height(self, lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        h = self._interp(np.stack([lat.ravel(), lon.ravel()], axis=-1)).reshape(lat.shape)  # type: ignore
        outside = ~np.isfinite(h)
        h = np.where(outside, self._fill, h) + self.above_ground_m
        return h, np.where(outside, 100.0, self.sigma)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, lat=self.lat, lon=self.lon, ground=self.ground,
                            meta=json.dumps({"sigma": self.sigma, "source": self.source,
                                             "attribution": self.attribution}))

    @classmethod
    def load(cls, path: Path) -> GridElevation:
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        return cls(z["lat"], z["lon"], z["ground"], meta["sigma"], meta["source"],
                   attribution=meta.get("attribution", ""))


# ---------------------------------------------------------------------------
# Copernicus GLO-30 + EGM2008
# ---------------------------------------------------------------------------


def glo30_tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{-lat_floor:02d}"
    ew = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{-lon_floor:03d}"
    return f"{ns}_00_{ew}_00"


def _target_grid(bbox: tuple[float, float, float, float], res_m: float) -> tuple[FloatArray, FloatArray]:
    lat_min, lon_min, lat_max, lon_max = bbox
    dlat = res_m / 111_320.0
    dlon = res_m / (111_320.0 * math.cos(math.radians(min(85.0, max(abs(lat_min), abs(lat_max))))))
    lat = np.arange(lat_min, lat_max + dlat, dlat)
    lon = np.arange(lon_min, lon_max + dlon, dlon)
    return lat, lon


def _read_into(url: str, lat: FloatArray, lon: FloatArray, out: FloatArray) -> bool:
    """Resample the part of a (remote) EPSG:4326 raster covering the target cell
    centres into ``out`` (lat ascending). Returns False if the file is missing."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError
    from rasterio.windows import from_bounds

    try:
        ds = rasterio.open(url if "://" not in url else f"/vsicurl/{url}")
    except RasterioIOError:
        return False
    with ds:
        b = ds.bounds
        dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
        rows = np.flatnonzero((lat >= b.bottom) & (lat < b.top))
        cols = np.flatnonzero((lon >= b.left) & (lon < b.right))
        if rows.size == 0 or cols.size == 0:
            return True
        win = from_bounds(lon[cols[0]] - dlon / 2, lat[rows[0]] - dlat / 2, lon[cols[-1]] + dlon / 2,
                          lat[rows[-1]] + dlat / 2, transform=ds.transform)
        data = ds.read(1, window=win, out_shape=(rows.size, cols.size), resampling=Resampling.bilinear,
                       boundless=True, fill_value=np.nan).astype(float)
        if ds.nodata is not None:
            data[data == ds.nodata] = np.nan
        out[np.ix_(rows, cols)] = data[::-1]  # raster rows run north -> south
    return True


def copernicus_dem(bbox: tuple[float, float, float, float], res_m: float = 100.0,
                   cache_dir: Path | str | None = None) -> GridElevation:
    """Ellipsoidal ground height over ``bbox`` (lat_min, lon_min, lat_max, lon_max)
    from GLO-30 + EGM2008, read remotely (COG overviews, so only ~a few MB move).
    Sea tiles (absent from GLO-30) count as 0 m orthometric. Cached as .npz."""
    try:
        import rasterio
    except ImportError as e:
        raise RuntimeError("the DEM needs rasterio: pip install 'dopplergeo[dem]'") from e

    key = hashlib.sha1(json.dumps([list(map(lambda x: round(x, 5), bbox)), res_m]).encode()).hexdigest()[:16]
    path = None
    if cache_dir is not None:
        d = Path(cache_dir) / "elevation"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"glo30_egm08_{key}.npz"
        if path.exists():
            return GridElevation.load(path)

    lat, lon = _target_grid(bbox, res_m)
    ortho = np.full((lat.size, lon.size), np.nan)
    geoid = np.full_like(ortho, np.nan)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1")
    with rasterio.Env(**env):
        for la in range(math.floor(lat[0]), math.floor(lat[-1]) + 1):
            for lo in range(math.floor(lon[0]), math.floor(lon[-1]) + 1):
                if not _read_into(GLO30_URL.format(tile=glo30_tile_name(la, lo)), lat, lon, ortho):
                    sel = np.ix_((lat >= la) & (lat < la + 1), (lon >= lo) & (lon < lo + 1))
                    ortho[sel] = 0.0  # no tile: open sea
        if not _read_into(EGM2008_URL, lat, lon, geoid) or not np.isfinite(geoid).all():
            raise RuntimeError("could not read the EGM2008 geoid grid from cdn.proj.org")
    model = GridElevation(lat, lon, ortho + geoid, GLO30_SIGMA_M, "copernicus-glo30+egm2008",
                          attribution=ATTRIBUTION_GLO30)
    if path is not None:
        model.save(path)
    return model


def _cells(bbox: tuple[float, float, float, float], res_m: float) -> int:
    lat, lon = _target_grid(bbox, res_m)
    return lat.size * lon.size


def regional_dem(bbox: tuple[float, float, float, float], cache_dir: Path | str | None = None,
                 max_cells: int = MAX_DEM_CELLS, fallback: ElevationModel | None = None,
                 max_coarse_res_m: float = MAX_COARSE_RES_M) -> GridElevation | PatchedDem:
    """The DEM for a search region. Small regions: one 100 m grid (as before).
    Larger ones: a ``PatchedDem`` whose coarse grid stays under ``max_cells``.
    Regions so large that the coarse grid would be coarser than
    ``max_coarse_res_m`` (hundreds of remote 1-degree tiles to open, e.g. a box
    over Europe) use ``fallback`` for regional queries instead (ADS-B airport
    ground or an unknown height, as the worldwide search does); the 100 m
    patches still serve the solver's local work."""
    if _cells(bbox, FINE_RES_M) <= max_cells:
        return copernicus_dem(bbox, FINE_RES_M, cache_dir=cache_dir)
    res = FINE_RES_M
    while _cells(bbox, res) > max_cells:
        res *= 1.25
    if res > max_coarse_res_m:
        coarse_model = fallback or ConstantElevation(0.0, 1500.0, "unknown")
        return PatchedDem(coarse_model, cache_dir=None if cache_dir is None else Path(cache_dir),
                          source=f"copernicus-glo30+egm2008 near the solution, {coarse_model.source} elsewhere")
    coarse = copernicus_dem(bbox, round(res, 1), cache_dir=cache_dir)
    # Interpolating across res-sized cells misses hills and valleys between samples.
    coarse.sigma = float(math.hypot(GLO30_SIGMA_M, 0.03 * res))
    coarse.source = f"copernicus-glo30+egm2008 ({res / 1000:.1f} km grid)"
    return PatchedDem(coarse, cache_dir=None if cache_dir is None else Path(cache_dir))


@dataclass
class PatchedDem:
    """A coarse regional grid plus 100 m patches fetched on demand.

    A query whose points fall into at most ``MAX_LOCAL_PATCHES`` patches (the
    solver refining one position, or probing an ellipse) gets 100 m heights;
    a wider one (the grid search over the region) gets the coarse grid with its
    larger sigma. A patch that cannot be read (offline) falls back to the coarse
    grid and is not retried."""

    coarse: ElevationModel
    cache_dir: Path | None = None
    source: str = "copernicus-glo30+egm2008"
    attribution: str = ATTRIBUTION_GLO30
    _patches: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _failed: set = field(default_factory=set, init=False, repr=False)

    def _patch(self, key: tuple[int, int]) -> GridElevation | None:
        if key in self._failed:
            return None
        if key in self._patches:
            self._patches.move_to_end(key)
            return self._patches[key]
        i, j = key
        # A small overlap so bilinear interpolation is defined up to the patch edge.
        pad = 0.002
        bbox = (i * PATCH_DEG - pad, j * PATCH_DEG - pad, (i + 1) * PATCH_DEG + pad, (j + 1) * PATCH_DEG + pad)
        try:
            g = copernicus_dem(bbox, FINE_RES_M, cache_dir=self.cache_dir)
        except Exception:
            self._failed.add(key)
            return None
        self._patches[key] = g
        while len(self._patches) > MAX_PATCHES_IN_MEMORY:
            self._patches.popitem(last=False)
        return g

    def height(self, lat: ArrayLike, lon: ArrayLike) -> tuple[FloatArray, FloatArray]:
        lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        h, s = self.coarse.height(lat, lon)
        h, s = np.array(h, dtype=float, ndmin=1).reshape(lat.shape), np.array(s, dtype=float).reshape(lat.shape)
        ki = np.floor(lat / PATCH_DEG).astype(np.int64)
        kj = np.floor(lon / PATCH_DEG).astype(np.int64)
        keys = set(zip(ki.ravel().tolist(), kj.ravel().tolist()))
        if len(keys) > MAX_LOCAL_PATCHES:
            return h, s
        for key in keys:
            g = self._patch(key)
            if g is None:
                continue
            m = (ki == key[0]) & (kj == key[1])
            hf, sf = g.height(lat[m], lon[m])
            h[m], s[m] = hf, sf
        return h, s
