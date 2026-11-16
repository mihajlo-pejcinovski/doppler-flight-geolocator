from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core import elevation as E
from dopplergeo.core.doppler import fit_doppler, frequency_from_trajectory
from dopplergeo.core.elevation import PHONE_HEIGHT_M, ConstantElevation, GridElevation, glo30_tile_name
from dopplergeo.core.geo import enu_to_geodetic, geodetic_to_enu, haversine_m
from dopplergeo.core.solver import FamilyObs, SolverSettings, solve
from dopplergeo.core.synthetic import StraightLineFlight, adsb_samples
from dopplergeo.flights import BBox, FlightTrack, TrackPoint, points_to_table
from dopplergeo.flights import ground as G


def test_constant_and_grid_models(tmp_path):
    h, s = ConstantElevation(12.0, 7.0).height([1.0, 2.0], [3.0, 4.0])
    assert h.tolist() == [12.0, 12.0] and s.tolist() == [7.0, 7.0]
    lat, lon = np.array([52.0, 53.0]), np.array([4.0, 5.0])
    g = GridElevation(lat, lon, np.array([[0.0, 10.0], [20.0, 30.0]]), sigma=6.0, source="t")
    h, s = g.height([52.5, 60.0], [4.5, 4.5])
    assert h[0] == pytest.approx(15.0 + PHONE_HEIGHT_M)  # bilinear centre
    assert s[0] == 6.0 and s[1] == 100.0  # outside the grid: median, wide sigma
    g.save(tmp_path / "g.npz")
    assert GridElevation.load(tmp_path / "g.npz").height(52.5, 4.5)[0] == pytest.approx(h[0])


def test_glo30_tile_names():
    assert glo30_tile_name(52, 4) == "N52_00_E004_00"
    assert glo30_tile_name(-1, -71) == "S01_00_W071_00"


def ground_rows(lat, lon, h, n, icao="aaaaaa"):
    return [TrackPoint(1e9 + i, icao, None, lat + 1e-4 * (i % 7), lon, h + (25 * 0.3048) * (i % 3 - 1),
                       None, 5.0, 0.0, 0.0, True) for i in range(n)]


def test_adsb_ground_sites_antenna_correction_and_sigma():
    rows = ground_rows(52.31, 4.76, 45.7, 200) + ground_rows(51.95, 4.44, 20.0, 80, "bbbbbb")
    rows += [TrackPoint(1e9, "cccccc", None, 52.311, 4.761, -518.0, None, 0.0, 0.0, 0.0, True)]  # junk
    rows += [TrackPoint(1e9, "dddddd", None, 52.0, 4.0, 3000.0, None, 200.0, 0.0, 0.0, False)]  # airborne
    m = G.adsb_ground_elevation(points_to_table(rows))
    assert m is not None and len(m.site_lat) == 2
    h, s = m.height([52.31, 51.95, 52.31], [4.76, 4.44, 5.76])
    assert h[0] == pytest.approx(45.7 - G.ADSB_ANTENNA_HEIGHT_M + PHONE_HEIGHT_M, abs=0.5)
    assert h[1] == pytest.approx(20.0 - G.ADSB_ANTENNA_HEIGHT_M + PHONE_HEIGHT_M, abs=0.5)
    assert s[0] == pytest.approx(m.base_sigma_m, abs=0.5) and s[2] > 60.0  # 68 km away
    assert G.adsb_ground_elevation(points_to_table(rows[-1:])) is None


def test_observer_elevation_fallback_chain(monkeypatch):
    region = BBox.around(52.3, 4.76, 10)
    table = points_to_table(ground_rows(52.31, 4.76, 45.7, 100))
    monkeypatch.setattr(G, "regional_dem", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    m, notes = G.observer_elevation("auto", region, table)
    assert m.source == "adsb-ground" and "offline" in notes[0]
    m, notes = G.observer_elevation("auto", region, None, constant_h=7.0)
    assert m.source == "constant" and m.height(0, 0)[0] == 7.0 and len(notes) == 2
    with pytest.raises(RuntimeError):
        G.observer_elevation("dem", region, table)


def test_copernicus_dem_from_local_tiles(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    def write(path, west, north, nx, ny, res_x, res_y, data):
        with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(west, north, res_x, res_y)) as ds:
            ds.write(data.astype("float32"), 1)

    # One land tile N52 E004 sloping west->east 0..100 m; E005 is "sea" (missing).
    nx, ny = 120, 120
    write(tmp_path / "N52_00_E004_00.tif", 4.0, 53.0, nx, ny, 1 / nx, 1 / ny,
          np.tile(np.linspace(0, 100, nx), (ny, 1)))
    write(tmp_path / "geoid.tif", 3.0, 54.0, 30, 30, 0.1, 0.1, np.full((30, 30), 43.0))
    monkeypatch.setattr(E, "GLO30_URL", str(tmp_path / "{tile}.tif"))
    monkeypatch.setattr(E, "EGM2008_URL", str(tmp_path / "geoid.tif"))
    m = E.copernicus_dem((52.2, 4.4, 52.4, 5.2), res_m=200.0, cache_dir=tmp_path / "cache")
    h, s = m.height([52.3, 52.3, 52.3], [4.0 + 0.5, 4.95, 5.1])
    assert h[0] == pytest.approx(50.0 + 43.0 + PHONE_HEIGHT_M, abs=2.0)
    assert h[1] == pytest.approx(95.0 + 43.0 + PHONE_HEIGHT_M, abs=2.0)
    assert h[2] == pytest.approx(0.0 + 43.0 + PHONE_HEIGHT_M, abs=0.1)  # sea tile
    assert s[0] == E.GLO30_SIGMA_M
    cached = list((tmp_path / "cache" / "elevation").glob("*.npz"))
    assert len(cached) == 1


def test_regional_dem_is_capped_and_patched(tmp_path, monkeypatch):
    """A region too large for one 100 m grid: a capped coarse grid for regional
    queries, 100 m patches for local ones (memory stays bounded)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    nx = ny = 1200  # 3 arc-seconds: an 800 m wide, 200 m high ridge at lon 4.505 the coarse grid smooths out
    lon = 4.0 + (np.arange(nx) + 0.5) / nx
    ridge = np.where(np.abs(lon - 4.505) < 0.006, 200.0, 0.0)
    with rasterio.open(tmp_path / "N52_00_E004_00.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(4.0, 53.0, 1 / nx, 1 / ny)) as ds:
        ds.write(np.tile(ridge, (ny, 1)).astype("float32"), 1)
    with rasterio.open(tmp_path / "geoid.tif", "w", driver="GTiff", width=30, height=30, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(3.0, 54.0, 0.1, 0.1)) as ds:
        ds.write(np.full((30, 30), 43.0, dtype="float32"), 1)
    monkeypatch.setattr(E, "GLO30_URL", str(tmp_path / "{tile}.tif"))
    monkeypatch.setattr(E, "EGM2008_URL", str(tmp_path / "geoid.tif"))

    bbox = (52.1, 4.1, 52.9, 4.9)
    assert isinstance(E.regional_dem(bbox, cache_dir=tmp_path / "c0"), E.GridElevation)  # small: as before
    m = E.regional_dem(bbox, cache_dir=tmp_path / "c1", max_cells=2_000, max_coarse_res_m=1e9)
    assert isinstance(m, E.PatchedDem)
    assert m.coarse.ground.size <= 2_000 and m.coarse.sigma > E.GLO30_SIGMA_M
    # Local query (one position): 100 m patch, sees the ridge.
    assert float(m.coarse.height(52.5, 4.505)[0]) < 150.0  # the coarse grid smooths the ridge away
    h, s = m.height(52.5, 4.505)
    assert float(h) == pytest.approx(200.0 + 43.0 + E.PHONE_HEIGHT_M, abs=5.0)
    assert float(s) == E.GLO30_SIGMA_M
    # Regional query (a grid over the box): coarse values, coarse sigma, no patch loads.
    lat, lon = np.meshgrid(np.linspace(52.15, 52.85, 30), np.linspace(4.15, 4.85, 30))
    n_before = len(m._patches)
    h, s = m.height(lat, lon)
    assert h.shape == lat.shape and np.all(s == m.coarse.sigma)
    assert len(m._patches) == n_before
    # A region too large for a coarse grid: the fallback serves regional queries, patches local ones.
    big = E.regional_dem(bbox, cache_dir=tmp_path / "c2", max_cells=2_000, max_coarse_res_m=150.0,
                         fallback=E.ConstantElevation(7.0, 1500.0, "unknown"))
    assert isinstance(big, E.PatchedDem) and isinstance(big.coarse, E.ConstantElevation)
    h, s = big.height(lat, lon)
    assert np.all(h == 7.0) and np.all(s == 1500.0)
    h, s = big.height(52.5, 4.505)
    assert float(h) == pytest.approx(200.0 + 43.0 + E.PHONE_HEIGHT_M, abs=5.0)
    # An unreadable patch falls back to the coarse grid.
    monkeypatch.setattr(E, "GLO30_URL", str(tmp_path / "missing" / "{tile}.tif"))
    monkeypatch.setattr(E, "EGM2008_URL", str(tmp_path / "missing.tif"))
    h, s = m.height(52.75, 4.75)
    assert float(s) == m.coarse.sigma


# ---------------------------------------------------------------------------
# Solver: observer on a 300 m hill
# ---------------------------------------------------------------------------

LAT0, LON0 = 47.0, 8.0
EPOCH = datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc).timestamp()
REGION = BBox.around(LAT0, LON0, 15.0)
H_TRUE = 300.0
OBS_LAT, OBS_LON, _ = enu_to_geodetic([2500.0, -1500.0, 0.0], LAT0, LON0)
OBS = geodetic_to_enu(OBS_LAT, OBS_LON, H_TRUE, LAT0, LON0)


def scenario():
    planes = [("4b1801", StraightLineFlight.from_cpa(150.0, 1800.0, 40.0, 60.0, 1500.0, observer=OBS)),
              ("4b1802", StraightLineFlight.from_cpa(120.0, 2400.0, 105.0, 170.0, 2000.0, observer=OBS)),
              ("4b1803", StraightLineFlight.from_cpa(170.0, 2600.0, 160.0, 290.0, 2300.0, observer=OBS))]
    fams, tracks = [], []
    for i, ((icao, fl), (t0, t1)) in enumerate(zip(planes, [(5, 80), (70, 150), (125, 200)])):
        t = np.arange(t0, t1, 0.1)
        f, _ = frequency_from_trajectory(t, 90.0 + 25 * i, fl.position, fl.velocity, OBS)
        sig = np.full(t.size, 0.03)
        fams.append(FamilyObs(i, t, f, np.ones(t.size), sig, fit_doppler(t, f, sigma=sig)))
        smp = adsb_samples(fl.position, fl.velocity, np.arange(-300, 500, 2.0), LAT0, LON0)
        smp["t"] = smp["t"] + EPOCH
        tracks.append(FlightTrack.from_arrays(icao, None, **smp))
    return fams, tracks


def err(sol):
    return float(haversine_m(sol.lat, sol.lon, OBS_LAT, OBS_LON))


def test_terrain_height_from_model_locates_exactly():
    fams, tracks = scenario()
    dem = ConstantElevation(H_TRUE, 6.0, "dem")
    sol = solve(fams, tracks, EPOCH, REGION, SolverSettings(elevation=dem))
    assert err(sol) < 5.0 and abs(sol.h - H_TRUE) < 2.0
    assert sol.h_source == "dem" and sol.h_fitted


def test_wrong_height_biases_position_unless_fitted():
    fams, tracks = scenario()
    wrong = ConstantElevation(0.0, 400.0, "guess")  # 300 m too low, honest wide sigma
    fixed = solve(fams, tracks, EPOCH, REGION, SolverSettings(elevation=wrong, fit_height=False))
    fitted = solve(fams, tracks, EPOCH, REGION, SolverSettings(elevation=wrong))
    # Held 300 m too low, the three look angles cannot agree: the solver keeps
    # fewer aircraft and lands in the wrong place.
    assert len(fixed.aircraft_used) < 3 and err(fixed) > 100.0
    # Fitted (auto, >= 2 aircraft), the height and the position both come back.
    assert fitted.h_fitted and abs(fitted.h - H_TRUE) < 10.0
    assert len(fitted.aircraft_used) == 3 and err(fitted) < 20.0
