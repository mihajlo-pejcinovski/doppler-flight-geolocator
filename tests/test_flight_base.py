from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core.doppler import frequency_from_trajectory
from dopplergeo.core.geo import ecef_to_enu, ecef_to_geodetic, enu_rotation, enu_to_ecef, geodetic_to_ecef
from dopplergeo.flights import BBox, FlightQuery, FlightTrack, TrackPoint, points_to_table, query_window
from dopplergeo.flights.base import table_to_points, tracks_from_table

LAT0, LON0 = 52.30, 4.76
T0 = datetime(2025, 12, 30, 14, 0, tzinfo=timezone.utc).timestamp()


def straight_flight(step=4.0, n=60, v_enu=(150.0, 150.0, 4.0), start_enu=(-20_000.0, -18_000.0, 2500.0),
                    with_velocity=True):
    """A constant-velocity ECEF line sampled like ADS-B (lat/lon/alt, gs/track/vrate)."""
    t = T0 + np.arange(n) * step
    v_ecef = np.asarray(v_enu) @ enu_rotation(LAT0, LON0)
    p0 = enu_to_ecef(start_enu, LAT0, LON0)
    pos = p0 + (t - T0)[:, None] * v_ecef
    lat, lon, h = ecef_to_geodetic(pos)
    # Local ENU velocity at each point (the line's direction relative to the local frame).
    loc = np.array([enu_rotation(a, b) @ v_ecef for a, b in zip(lat, lon)])
    gs = np.hypot(loc[:, 0], loc[:, 1])
    trk = np.degrees(np.arctan2(loc[:, 0], loc[:, 1])) % 360
    pts = [TrackPoint(float(t[i]), "abc123", "KLM1" if i > 3 else None, float(lat[i]), float(lon[i]),
                      float(h[i]), float(h[i]) - 80.0,
                      float(gs[i]) if with_velocity else None, float(trk[i]) if with_velocity else None,
                      float(loc[i, 2]) if with_velocity else None)
           for i in range(n)]
    truth = (lambda tt: p0 + (np.asarray(tt) - T0)[:, None] * v_ecef, lambda tt: np.tile(v_ecef, (len(tt), 1)))
    return pts, truth


def test_bbox_geometry():
    b = BBox.around(LAT0, LON0, 50.0)
    assert b.lat_max - b.lat_min == pytest.approx(2 * 50 / 111.32)
    assert b.covers(BBox.around(LAT0, LON0, 10.0)) and not BBox.around(LAT0, LON0, 10.0).covers(b)
    assert b.contains(LAT0, LON0) and not b.contains(LAT0 + 1, LON0)
    assert BBox.parse("1,2,3,4") == BBox(1, 2, 3, 4)
    with pytest.raises(ValueError):
        BBox(3, 0, 1, 1)


def test_query_window_pads_time_and_area():
    region = BBox.around(LAT0, LON0, 50)
    q = query_window(T0, 300.0, region, clock_bound_s=30.0)
    assert q.t0 == T0 - 150 and q.t1 == T0 + 450
    assert q.bbox.covers(region.expanded(19.9))


def test_utc_dates_span_midnight():
    mid = datetime(2025, 12, 31, tzinfo=timezone.utc).timestamp()
    q = FlightQuery(mid - 60, mid + 60, BBox.world())
    assert [d.day for d in q.utc_dates()] == [30, 31]
    assert len(FlightQuery(mid + 10, mid + 20, BBox.world()).utc_dates()) == 1


def test_table_round_trip():
    pts, _ = straight_flight(n=5)
    back = table_to_points(points_to_table(pts))
    for a, b in zip(pts, back, strict=True):
        assert (a.t, a.icao24, a.callsign, a.lat, a.lon) == (b.t, b.icao24, b.callsign, b.lat, b.lon)
        assert b.alt_geo_m == pytest.approx(a.alt_geo_m, rel=1e-6)  # float32 columns
        assert b.gs_mps == pytest.approx(a.gs_mps, rel=1e-6)


def test_hermite_track_reproduces_straight_line():
    pts, (pos, vel) = straight_flight()
    (tr,) = tracks_from_table(points_to_table(pts))
    assert tr.callsign == "KLM1" and tr.alt_source == "geo"
    tq = np.linspace(tr.t_start, tr.t_end, 777)
    assert np.abs(tr.position(tq) - pos(tq)).max() < 0.05  # metres
    assert np.abs(tr.velocity(tq) - vel(tq)).max() < 0.01  # m/s
    assert np.isnan(tr.position([tr.t_end + 5.0])).all()


def test_finite_difference_velocity_when_not_reported():
    pts, (pos, vel) = straight_flight(with_velocity=False)
    (tr,) = tracks_from_table(points_to_table(pts))
    tq = np.linspace(tr.t_start, tr.t_end, 500)
    assert np.abs(tr.position(tq) - pos(tq)).max() < 0.5
    assert np.abs(tr.velocity(tq) - vel(tq)).max() < 0.5


def test_trajectory_drives_doppler_model():
    """A FlightTrack plugs into the forward model and matches the analytic trajectory."""
    pts, (pos, vel) = straight_flight(step=2.0, n=120)
    (tr,) = tracks_from_table(points_to_table(pts))
    obs = geodetic_to_ecef(LAT0, LON0, 0.0)
    # Reception times whose emission time lies inside the track (delays here are up to ~80 s).
    t = np.linspace(tr.t_start + 120, tr.t_end - 10, 200)
    f_tr, _ = frequency_from_trajectory(t, 100.0, tr.position, tr.velocity, obs)
    f_true, _ = frequency_from_trajectory(t, 100.0, pos, vel, obs)
    assert np.abs(f_tr - f_true).max() < 1e-4
    assert f_tr[0] > 100 > f_tr[-1]


def test_reliability_mask_and_alt_offset():
    pts, _ = straight_flight(n=30)
    # Drop reports 10..19 (40 s hole) and remove geometric altitude from half the rest.
    pts = [p for i, p in enumerate(pts) if not 10 <= i < 20]
    pts = [p if i % 2 else TrackPoint(**{**p.__dict__, "alt_geo_m": None}) for i, p in enumerate(pts)]
    (tr,) = tracks_from_table(points_to_table(pts))
    assert tr.alt_source == "baro+offset"
    assert np.allclose(tr.alt_m, [p.alt_baro_m + 80.0 for p in pts], atol=1e-3)
    hole = T0 + 4 * 14.5
    assert not tr.reliable([hole])[0]
    assert tr.reliable([T0 + 4 * 3.5, T0 + 4 * 25.5]).all()
    assert not tr.reliable([T0 - 1.0])[0]
    grid = tr.resample(1.0)
    assert grid["ecef"].shape == (grid["t"].size, 3) and (~grid["reliable"]).sum() >= 39


def test_split_by_aircraft_gap_and_ground():
    a, _ = straight_flight(n=10)
    b = [TrackPoint(**{**p.__dict__, "icao24": "def456", "callsign": "X"}) for p in a]
    late = [TrackPoint(**{**p.__dict__, "t": p.t + 3600}) for p in a]
    ground = [TrackPoint(**{**p.__dict__, "icao24": "777777", "on_ground": True}) for p in a]
    tracks = tracks_from_table(points_to_table(a + b + late + ground))
    assert sorted((t.icao24, len(t)) for t in tracks) == [("abc123", 10), ("abc123", 10), ("def456", 10)]
    assert len(tracks_from_table(points_to_table(ground), include_ground=True)) == 1
    assert ecef_to_enu(tracks[0].position([tracks[0].t_start]), LAT0, LON0).shape == (1, 3)


def test_aircraft_without_usable_altitude_is_skipped():
    """Real adsb.lol data: an airborne row with null altitude must not crash track building."""
    pts, _ = straight_flight(n=5)
    bad = [TrackPoint(**{**pts[0].__dict__, "icao24": "nul000", "alt_geo_m": None, "alt_baro_m": None})]
    single = [TrackPoint(**{**pts[0].__dict__, "icao24": "one000"})]
    tracks = tracks_from_table(points_to_table(pts + bad + single))
    assert [t.icao24 for t in tracks] == ["abc123"]
