"""Worldwide search (no region hint) on a synthetic world with decoys."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core.elevation import ConstantElevation
from dopplergeo.core.doppler import fit_doppler, frequency_from_trajectory
from dopplergeo.core.geo import enu_to_geodetic, geodetic_to_enu, haversine_m
from dopplergeo.core.solver import FamilyObs, SolverSettings
from dopplergeo.core.synthetic import StraightLineFlight, adsb_samples
from dopplergeo.flights import FlightTrack
from dopplergeo.global_search import GlobalSearchError, solve_global

EPOCH = datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc).timestamp()
HOME = (52.30, 4.76)
TWIN_HOME = (40.64, -73.78)  # an observer near JFK would hear the same curve from the twin


def observer(lat0, lon0):
    la, lo, _ = enu_to_geodetic([1800.0, -900.0, 0.0], lat0, lon0)
    return float(la), float(lo), geodetic_to_enu(la, lo, 0.0, lat0, lon0)


def track(icao, flight, lat0, lon0):
    s = adsb_samples(flight.position, flight.velocity, np.arange(-150.0, 300.0, 2.0), lat0, lon0)
    s["t"] = s["t"] + EPOCH
    return FlightTrack.from_arrays(icao, None, **s)


def family(fid, flight, obs, t0, t1, f0):
    t = np.arange(t0, t1, 0.1)
    f, _ = frequency_from_trajectory(t, f0, flight.position, flight.velocity, obs)
    sig = np.full(t.size, 0.03)
    return FamilyObs(fid, t, f, np.ones(t.size), sig, fit_doppler(t, f, sigma=sig))


@pytest.fixture(scope="module")
def world():
    lat, lon, obs = observer(*HOME)
    a = StraightLineFlight.from_cpa(95.0, 1400.0, 40.0, 70.0, 900.0, observer=obs)
    b = StraightLineFlight.from_cpa(120.0, 1900.0, 100.0, 200.0, 1300.0, observer=obs)
    _, _, obs_twin = observer(*TWIN_HOME)
    twin = StraightLineFlight.from_cpa(95.0, 1400.0, 40.0, 70.0, 900.0, observer=obs_twin)
    tracks = [track("4840d6", a, *HOME), track("3c6444", b, *HOME), track("a1b2c3", twin, *TWIN_HOME)]
    rng = np.random.default_rng(7)
    for i in range(300):  # low decoys all over the world, plausible speeds
        lat0, lon0 = rng.uniform(-50, 60), rng.uniform(-150, 150)
        fl = StraightLineFlight.from_cpa(rng.uniform(70, 150), rng.uniform(800, 3000), rng.uniform(0, 150),
                                         rng.uniform(0, 360), None, observer=(0.0, 0.0, 0.0))
        tracks.append(track(f"d{i:05x}", fl, lat0, lon0))
    fams = [family(0, a, obs, 5.0, 80.0, 220.0), family(1, b, obs, 60.0, 150.0, 300.0)]
    return fams, tracks, (lat, lon)


def test_two_aircraft_unique_worldwide(world):
    fams, tracks, truth = world
    res = solve_global(fams, tracks, EPOCH, SolverSettings())
    assert res.solution is not None and not res.ambiguous
    assert haversine_m(res.solution.lat, res.solution.lon, *truth) < 50.0
    assert set(res.solution.aircraft_used) == {"4840d6", "3c6444"}
    assert res.solution.grade in ("A", "B")
    assert res.candidates_per_family[0] < len(tracks)  # the screen removed most of the world


def test_one_aircraft_with_a_twin_is_ambiguous(world):
    fams, tracks, truth = world
    twin_lat, twin_lon, _ = observer(*TWIN_HOME)
    # Ground height known roughly (as locate provides from airport ADS-B); with no
    # terrain information at all a single straight pass leaves a ~3 km band instead.
    res = solve_global(fams[:1], tracks, EPOCH, SolverSettings(), ground=ConstantElevation(0.0, 10.0, "airport"))
    assert res.ambiguous and res.solution.grade == "C"
    places = res.places()
    near = lambda p, q: haversine_m(p["lat"], p["lon"], *q) < 100.0  # noqa: E731
    assert any(near(p, truth) for p in places) and any(near(p, (twin_lat, twin_lon)) for p in places)


def test_no_matching_aircraft(world):
    fams, tracks, _ = world
    res = solve_global(fams, tracks[3:], EPOCH, SolverSettings())  # only decoys
    assert res.solution is None or res.solution.grade == "C"


def test_needs_a_reliable_family(world):
    fams, tracks, _ = world
    f = fams[0]
    bad = FamilyObs(9, f.t, f.f, f.k, f.sigma, f.fit)
    bad.loose_override = True
    with pytest.raises(GlobalSearchError):
        solve_global([bad], tracks, EPOCH, SolverSettings())
