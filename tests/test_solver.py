"""M4 solver: synthetic end-to-end scenarios with a known observer.

Aircraft fly analytic trajectories in the ENU frame at the region centre. They
are 'logged' as ADS-B (lat/lon/alt, gs/track/vrate every 2 s) and rebuilt into
FlightTracks, while the measured tones come from the exact analytic
trajectory. So the solver sees the same information as with real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
import pytest

from dopplergeo.core.doppler import fit_doppler, frequency_from_trajectory
from dopplergeo.core.geo import enu_to_geodetic, geodetic_to_enu, haversine_m
from dopplergeo.core.solver import FamilyObs, SolverError, SolverSettings, grade_solution, solve
from dopplergeo.core.synthetic import StraightLineFlight, TurningFlight, adsb_samples
from dopplergeo.flights import BBox, FlightTrack

LAT0, LON0 = 52.30, 4.76
EPOCH = datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc).timestamp()
REGION = BBox.around(LAT0, LON0, 15.0)
OBS_EN = (3200.0, -1800.0)
OBS_LAT, OBS_LON, _ = enu_to_geodetic([OBS_EN[0], OBS_EN[1], 0.0], LAT0, LON0)
OBS = geodetic_to_enu(OBS_LAT, OBS_LON, 0.0, LAT0, LON0)  # exact Cartesian (curvature included)


@dataclass
class Plane:
    icao: str
    flight: object  # .position/.velocity in ENU, relative seconds


def straight(icao, speed, heading, d, tau0, alt):
    return Plane(icao, StraightLineFlight.from_cpa(speed, d, tau0, heading, alt, observer=OBS))


def to_track(p: Plane, t0=-300.0, t1=500.0, step=2.0) -> FlightTrack:
    s = adsb_samples(p.flight.position, p.flight.velocity, np.arange(t0, t1, step), LAT0, LON0)
    s["t"] = s["t"] + EPOCH
    return FlightTrack.from_arrays(p.icao, p.icao.upper(), **s)


def family(fid, p: Plane, t0, t1, f0=95.0, ks=(1, 2), sigma=0.03, noise=0.0, clock_err=0.0, seed=0):
    """Tones heard at true times t; the file clock reads t - clock_err."""
    rng = np.random.default_rng(seed)
    t_true = np.arange(t0, t1, 0.1)
    ts, fs, kk = [], [], []
    for k in ks:
        f, _ = frequency_from_trajectory(t_true, f0, p.flight.position, p.flight.velocity, OBS, k=k)
        ts.append(t_true - clock_err)
        fs.append(f + rng.normal(0, noise, f.size) if noise else f)
        kk.append(np.full(f.size, float(k)))
    t, f, k = np.concatenate(ts), np.concatenate(fs), np.concatenate(kk)
    sig = np.full(t.size, sigma)
    return FamilyObs(fid, t, f, k, sig, fit_doppler(t, f, k=k, sigma=sig))


def err_m(sol_lat, sol_lon):
    return float(haversine_m(sol_lat, sol_lon, OBS_LAT, OBS_LON))


A = straight("4840d6", 150.0, 60.0, 1800.0, 40.0, 1500.0)
B = straight("3c6444", 120.0, 170.0, 2400.0, 105.0, 2000.0)
SLOW_FAR = straight("aaaa01", 150.0, 60.0, 1800.0, 40.0, 1500.0)
FAST = straight("aaaa02", 240.0, 250.0, 2500.0, 40.0, 2000.0)


def far_away(p: Plane, de=60_000.0) -> Plane:
    class Shifted:
        def position(self, t):
            return p.flight.position(t) + np.array([de, 0.0, 0.0])

        def velocity(self, t):
            return p.flight.velocity(t)
    return Plane("aaaa01", Shifted())


@pytest.fixture(scope="module")
def two_aircraft_solution():
    fams = [family(0, A, 5.0, 80.0), family(1, B, 70.0, 150.0, f0=140.0, ks=(1,))]
    tracks = [to_track(p) for p in (A, B, far_away(SLOW_FAR), FAST)]
    return fams, solve(fams, tracks, EPOCH, REGION.expanded(20), SolverSettings())


def test_two_aircraft_noise_free_locates_within_100_m(two_aircraft_solution):
    fams, sol = two_aircraft_solution
    assert err_m(sol.lat, sol.lon) < 100.0  # M4 checklist
    assert sol.assignment == {0: "4840d6", 1: "3c6444"}
    assert sol.aircraft_used == ["3c6444", "4840d6"]
    assert sol.resolved and sol.grade == "A"
    assert sol.ellipse_95[0] < 500.0
    assert sol.unexplained == []


def test_gating_drops_wrong_speed_and_far_aircraft(two_aircraft_solution):
    _, sol = two_aircraft_solution
    by_family = {}
    for c in sol.candidates:
        by_family.setdefault(c.family_id, set()).add(c.icao24)
    # The far-away twin and the fast jet never pass; B (120 m/s) is within 25% of
    # family 0's 150 m/s and nearby, so it is a legitimate candidate the grid must reject.
    assert by_family[0] == {"4840d6", "3c6444"} and by_family[1] <= {"3c6444", "4840d6"}
    assert not any(c.icao24 in ("aaaa01", "aaaa02") for c in sol.candidates)
    chi = {(c.family_id, c.icao24): c.chi2 for c in sol.candidates}
    assert chi[(0, "4840d6")] < 1e-3 * chi[(0, "3c6444")]


def test_grid_heatmap_and_predicted_curve(two_aircraft_solution):
    fams, sol = two_aircraft_solution
    lat, lon, d = sol.grid.heatmap()
    assert lat.shape == d.shape and np.nanmin(d) == 0.0
    j = np.nanargmin(d)
    assert haversine_m(lat.ravel()[j], lon.ravel()[j], OBS_LAT, OBS_LON) < 1000.0
    f = fams[0]
    pred = sol.predict(0, f.t, f.k)
    assert np.nanmax(np.abs(pred - f.f)) < 0.05  # the trust signal: curve sits on the data
    gj = sol.to_geojson()
    kinds = [ft["properties"]["kind"] for ft in gj["features"]]
    assert kinds[:2] == ["solution", "ellipse_95"]


def test_single_straight_pass_leaves_a_mirror():
    fams = [family(0, A, 5.0, 80.0)]
    sol = solve(fams, [to_track(A)], EPOCH, REGION, SolverSettings())
    assert not sol.resolved and sol.grade == "C"
    assert sol.mirror is not None
    best, mirror = err_m(sol.lat, sol.lon), err_m(sol.mirror["lat"], sol.mirror["lon"])
    assert min(best, mirror) < 100.0
    assert max(best, mirror) > 1000.0  # the other one is across the track


def test_curved_track_breaks_the_mirror():
    # Turning right at 1.5 deg/s through the pass; observer 1.4 km left of the turn.
    heading0, rate, v = 20.0, 1.5, 110.0
    probe = TurningFlight((0.0, 0.0, 900.0), v, heading0, rate)
    # Place the turn so its position at tau=40 s lies 1400 m to the left of the observer's... (shift).
    p40 = probe.position([40.0])[0]
    hd = math.radians(heading0 + rate * 40.0)
    left = np.array([-math.cos(hd), math.sin(hd)])
    target = np.array([OBS[0], OBS[1]]) + 1400.0 * -left
    p0 = (target[0] - p40[0], target[1] - p40[1], 900.0)
    plane = Plane("4ca7b1", TurningFlight(p0, v, heading0, rate))
    fams = [family(0, plane, 5.0, 80.0, f0=88.0)]
    sol = solve(fams, [to_track(plane)], EPOCH, REGION, SolverSettings())
    assert err_m(sol.lat, sol.lon) < 100.0
    assert sol.resolved, sol.mirror
    assert sol.grade == "B"


def test_clock_offset_is_recovered():
    fams = [family(0, A, 5.0, 80.0, clock_err=23.0), family(1, B, 70.0, 150.0, f0=140.0, ks=(1,),
                                                           clock_err=23.0)]
    s = SolverSettings(clock_bound_s=60.0)
    sol = solve(fams, [to_track(A), to_track(B)], EPOCH, REGION, s)
    assert abs(sol.dt_clock - 23.0) < 0.05
    assert err_m(sol.lat, sol.lon) < 100.0


# Manchester-like approach stream: same line, a plane every ~2.5 min, speeds 78-90 m/s.
APPROACH = [straight(f"bbbb0{i}", v, 235.0, 900.0, tc, 700.0)
            for i, (v, tc) in enumerate([(78.0, 60.0), (90.0, 220.0), (84.0, 380.0)])]


def _approach_case(n_passes, clock_err=-5.0, bound=600.0, **settings):
    fams = [family(i, p, p_tc - 35.0, p_tc + 35.0, f0=f0, ks=(1,), sigma=0.3, noise=0.3, clock_err=clock_err,
                   seed=i)
            for i, (p, p_tc, f0) in enumerate(zip(APPROACH[:n_passes], (60.0, 220.0, 380.0), (420.0, 300.0, 150.0)))]
    tracks = [to_track(p, -400.0, 900.0) for p in APPROACH]
    return solve(fams, tracks, EPOCH, REGION, SolverSettings(clock_bound_s=bound, **settings))


def _best_or_mirror_err(sol):
    errs = [err_m(sol.lat, sol.lon)]
    if sol.mirror is not None:
        errs.append(err_m(sol.mirror["lat"], sol.mirror["lon"]))
    return min(errs)


def test_passes_on_one_line_share_the_clock():
    """One level straight pass with a free clock is degenerate: shifting the clock
    by dt and the observer by v*dt along the track gives the same tone curve. Passes
    at different speeds only agree on one clock offset, which pins the position
    along the line (the across-line mirror of a shared straight line remains)."""
    one = _approach_case(1)
    assert _best_or_mirror_err(one) > 2000.0  # slid along the line
    sol = _approach_case(3)
    assert abs(sol.dt_clock - -5.0) < 2.0, sol.dt_clock
    assert _best_or_mirror_err(sol) < 300.0
    assert len(sol.aircraft_used) == 3


def test_noisy_tones_still_inside_ellipse():
    fams = [family(0, A, 5.0, 80.0, noise=0.05, sigma=0.05, seed=1),
            family(1, B, 70.0, 150.0, f0=140.0, ks=(1,), noise=0.05, sigma=0.05, seed=2)]
    sol = solve(fams, [to_track(A), to_track(B)], EPOCH, REGION, SolverSettings())
    a, _, _ = sol.ellipse_95
    assert err_m(sol.lat, sol.lon) < max(a, 100.0)
    assert 0.5 < sol.chi2_red < 2.0


def test_unexplained_family_is_reported():
    ghost = straight("ffffff", 260.0, 300.0, 3000.0, 60.0, 2500.0)
    fams = [family(0, A, 5.0, 80.0), family(7, B, 70.0, 150.0, f0=140.0, ks=(1,)),
            family(9, ghost, 20.0, 100.0, f0=60.0, ks=(1,))]
    sol = solve(fams, [to_track(A), to_track(B)], EPOCH, REGION, SolverSettings())
    assert sol.unexplained == [9]
    assert err_m(sol.lat, sol.lon) < 100.0


def test_no_candidates_raises():
    with pytest.raises(SolverError):
        solve([family(0, A, 5.0, 80.0)], [to_track(FAST)], EPOCH, REGION, SolverSettings())


def test_grade_rules():
    assert grade_solution(2, True, 400.0) == "A"
    assert grade_solution(1, True, 400.0) == "B"
    assert grade_solution(2, True, 1500.0) == "B"
    assert grade_solution(2, False, 100.0) == "C"
    assert grade_solution(2, True, 2500.0) == "C"
    assert grade_solution(2, True, 100.0, contradicted=1) == "C"  # a good family the position cannot explain


def test_clock_prior_pins_a_single_pass_along_its_track():
    """An original phone clock (prior N(0, 1 s)) removes the clock/along-track slide
    that leaves one straight pass kilometres off with a free clock."""
    sol = _approach_case(1, clock_err=-0.6, clock_sigma_s=1.0)
    assert _best_or_mirror_err(sol) < 300.0
    assert abs(sol.dt_clock - -0.6) < 1.5


def test_late_adsb_track_is_absorbed_by_its_lag():
    """One aircraft's ADS-B timestamps run 1 s late (position latency). Without
    timing slack the solver's ellipse excludes the truth by hundreds of sigma and a
    good pass is pruned; with a per-aircraft lag both passes are kept, the error
    drops and the ellipse is honest. (The two lags are not separable in this
    geometry: only their effect on the position is tested.)"""
    fams = [family(0, A, 5.0, 80.0, noise=0.05, sigma=0.05, seed=1),
            family(1, B, 70.0, 150.0, f0=140.0, ks=(1,), noise=0.05, sigma=0.05, seed=2)]
    on_time = to_track(B)
    tracks = [to_track(A), replace(on_time, t=on_time.t + 1.0)]

    def mahal2(sol):
        d = np.array(geodetic_to_enu(OBS_LAT, OBS_LON, 0.0, sol.frame.lat0, sol.frame.lon0)[:2]) \
            - np.array([sol.best.e, sol.best.n])
        return float(d @ np.linalg.solve(sol.cov_en, d))

    stiff = solve(fams, tracks, EPOCH, REGION, SolverSettings(adsb_lag_sigma_s=0.0))
    sol = solve(fams, tracks, EPOCH, REGION, SolverSettings(adsb_lag_sigma_s=1.0))
    assert mahal2(stiff) > 100.0
    assert mahal2(sol) < 5.99  # truth inside the 95% ellipse
    assert err_m(sol.lat, sol.lon) < err_m(stiff.lat, stiff.lon)
    # The stiff fit cannot reconcile the two timings and prunes a good pass.
    assert len(stiff.families_used) == 1 and sol.families_used == [0, 1]


def test_threaded_grid_matches_serial_and_cancels_promptly(two_aircraft_solution):
    """The grid search's (family, aircraft) pairs run on threads; the result must
    equal a one-thread scan, and an exception from ``progress`` (a cancelled GUI
    job) must stop it without finishing the queued pairs."""
    fams, sol = two_aircraft_solution
    tracks = [to_track(p) for p in (A, B, far_away(SLOW_FAR), FAST)]
    serial = solve(fams, tracks, EPOCH, REGION.expanded(20), SolverSettings(grid_workers=1))
    for k in ("chi2", "family_chi2", "assign", "dt"):
        assert np.array_equal(getattr(serial.grid, k), getattr(sol.grid, k), equal_nan=True), k
    assert (serial.lat, serial.lon) == (sol.lat, sol.lon)

    class Cancelled(Exception):
        pass

    calls = []

    def progress(stage, done, total):
        calls.append(stage)
        if stage == "grid search" and done >= 1:
            raise Cancelled()

    with pytest.raises(Cancelled):
        solve(fams, tracks, EPOCH, REGION.expanded(20), SolverSettings(grid_workers=4), progress=progress)
    assert calls.count("grid search") == 1
