"""Moving-air forward model: speed of sound with height, uniform wind.

The reference here is independent of the solver's Newton iteration: for each
emission time tau the reception time follows in closed form from the wavefront
geometry in the air's frame (a quadratic in the travel time s), and the received
frequency is f0 / (dt/dtau), differentiated numerically.
"""

from __future__ import annotations

import numpy as np
import pytest

from dopplergeo.core.atmosphere import (Atmosphere, path_mean_speed_of_sound, speed_of_sound,
                                        temperature_for_speed, wind_from_met)
from dopplergeo.core.doppler import fit_doppler
from dopplergeo.core.solver import FamilyObs, SampledTrajectory, SolverSettings, log_doppler, solve
from test_solver import EPOCH, LAT0, LON0, OBS, REGION, A, B, err_m, to_track


def reception(p: np.ndarray, obs: np.ndarray, tau: np.ndarray, c: float, w: np.ndarray) -> np.ndarray:
    """Reception time of sound emitted at tau from p (N, 3): |d - w s| = c s, d = obs - p."""
    d = obs - p
    wd = d @ w
    a = c * c - w @ w
    s = (-wd + np.sqrt(wd * wd + a * np.einsum("ij,ij->i", d, d))) / a
    return tau + s


def windy_tones(plane, t_rec: np.ndarray, f0: float, c: float, w: np.ndarray) -> np.ndarray:
    """Exact received frequency at reception times t_rec (constant c, uniform wind)."""
    tau = np.arange(t_rec.min() - 60.0, t_rec.max() + 1.0, 0.01)
    t = reception(plane.flight.position(tau), OBS, tau, c, w)
    f = f0 / np.gradient(t, tau)
    return np.interp(t_rec, t, f)


def test_c_path_matches_numeric_harmonic_mean():
    a = Atmosphere(temp_c=22.0)
    for dh in (0.0, 0.5, 300.0, 2500.0, 10_000.0):
        assert a.c_path(np.array(dh)) == pytest.approx(path_mean_speed_of_sound(0.0, dh, 22.0, n=4001), rel=1e-6)
    assert Atmosphere(temp_c=22.0, lapse=False).c_path(np.array(5000.0)) == pytest.approx(speed_of_sound(22.0))


def test_wind_from_met_convention():
    e, n = wind_from_met(10.0, 270.0)  # a westerly blows towards the east
    assert e == pytest.approx(10.0) and n == pytest.approx(0.0, abs=1e-9)


def _traj():
    return SampledTrajectory(to_track(A), LAT0, LON0, EPOCH)


def test_still_air_matches_the_old_model():
    traj = _traj()
    t = np.arange(10.0, 80.0, 0.5)
    c = 340.0
    old, ok0, _ = log_doppler(traj, OBS, t, c)
    new, ok1, _ = log_doppler(traj, OBS, t, c, atmos=Atmosphere(temp_c=temperature_for_speed(c), lapse=False))
    assert (ok0 == ok1).all()
    assert np.max(np.abs(new - old)[ok0]) < 1e-9


@pytest.mark.parametrize("wind", [(12.0, 0.0), (0.0, -9.0), (7.0, 7.0)])
def test_uniform_wind_matches_wavefront_geometry(wind):
    traj = _traj()
    atm = Atmosphere(temp_c=15.0, wind_e=wind[0], wind_n=wind[1], lapse=False)
    t = np.arange(15.0, 75.0, 0.5)
    lg, ok, _ = log_doppler(traj, OBS, t, 0.0, atmos=atm)
    ref = windy_tones(A, t, 1.0, atm.c_surface, atm.wind)
    assert ok.all()
    # The trajectory is a 2 s ADS-B resampling of the exact one; agreement to ~1e-6
    # (0.1 mHz at 100 Hz) checks the frequency formula, far below the wind effect (~1e-2).
    assert np.max(np.abs(np.exp(lg) - ref)) < 2e-6
    still, _, _ = log_doppler(traj, OBS, t, atm.c_surface)
    assert np.max(np.abs(np.exp(still) - ref)) > 1e-3


def _windy_family(fid, plane, t0, t1, f0, ks, c, w, sigma=0.03):
    t = np.arange(t0, t1, 0.1)
    ts, fs, kk = [], [], []
    for k in ks:
        ts.append(t)
        fs.append(k * windy_tones(plane, t, f0, c, w))
        kk.append(np.full(t.size, float(k)))
    t, f, k = np.concatenate(ts), np.concatenate(fs), np.concatenate(kk)
    sig = np.full(t.size, sigma)
    return FamilyObs(fid, t, f, k, sig, fit_doppler(t, f, k=k, sigma=sig))


def test_solver_with_the_right_wind_removes_the_bias():
    """Tones recorded in a 12 m/s wind: the still-air solver is biased, the solver
    given the wind is not, and fitting the wind from a calm prior recovers most of it."""
    atm = Atmosphere(temp_c=15.0, wind_e=-12.0, wind_n=4.0, lapse=False)
    c, w = atm.c_surface, atm.wind
    fams = [_windy_family(0, A, 5.0, 80.0, 95.0, (1, 2), c, w), _windy_family(1, B, 70.0, 150.0, 140.0, (1,), c, w)]
    tracks = [to_track(A), to_track(B)]
    base = dict(adsb_lag_sigma_s=0.0)
    still = solve(fams, tracks, EPOCH, REGION, SolverSettings(**base))
    right = solve(fams, tracks, EPOCH, REGION, SolverSettings(atmos=atm, **base))
    calm = Atmosphere(temp_c=15.0, lapse=False)
    fitted = solve(fams, tracks, EPOCH, REGION, SolverSettings(atmos=calm, fit_wind=True, wind_sigma_mps=10.0,
                                                               **base))
    e_still, e_right, e_fit = (err_m(s.lat, s.lon) for s in (still, right, fitted))
    assert e_right < 5.0, e_right
    assert e_still > 5 * e_right, (e_still, e_right)
    assert e_fit < 0.5 * e_still, (e_fit, e_still)
    wa = fitted.best.atmos
    assert np.hypot(wa.wind_e - atm.wind_e, wa.wind_n - atm.wind_n) < 4.0, (wa.wind_e, wa.wind_n)


def test_open_meteo_parsing_interpolation_and_cache(tmp_path):
    """Archive answer -> Atmosphere at 11:30 UTC (halfway between hours), wind
    interpolated as a vector; the second call is served from the cache."""
    import httpx

    from dopplergeo.weather import atmosphere_at

    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.url.params["latitude"] == "53.4" and req.url.params["longitude"] == "-2.3"
        hours = [f"2026-09-03T{h:02d}:00" for h in range(24)]
        temp = [10.0 + h for h in range(24)]
        return httpx.Response(200, json={"hourly": {
            "time": hours, "temperature_2m": temp,
            "wind_speed_100m": [8.0] * 24, "wind_direction_100m": [350.0 if h < 12 else 10.0 for h in range(24)],
            "wind_speed_10m": [4.0] * 24, "wind_direction_10m": [0.0] * 24}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    from datetime import datetime, timezone
    t = datetime(2026, 9, 3, 11, 30, tzinfo=timezone.utc).timestamp()
    a = atmosphere_at(53.3537, -2.2750, t, tmp_path, client=client)
    assert a.temp_c == pytest.approx(21.5)
    # 350 deg and 10 deg average to a northerly (from 0 deg): blowing south.
    assert a.wind_e == pytest.approx(0.0, abs=1e-9) and a.wind_n == pytest.approx(-8.0 * np.cos(np.radians(10.0)))
    assert "archive" in a.source
    atmosphere_at(53.3537, -2.2750, t, tmp_path, client=client)
    assert len(calls) == 1


@pytest.mark.parametrize("atmos", [None, Atmosphere(temp_c=15.0, lapse=False),
                                   Atmosphere(temp_c=24.0, wind_e=-9.0, wind_n=6.0, lapse=True)])
@pytest.mark.parametrize("parallel_min", [0, 10**9])  # the threaded and the serial loop
def test_compiled_forward_model_matches_numpy(atmos, parallel_min, monkeypatch):
    """The numba kernel against the NumPy reference: a grid of observers x times
    (broadcast like the grid search), including times outside the track (invalid)."""
    from dopplergeo.core import forward
    from dopplergeo.core.solver import log_doppler_numpy

    monkeypatch.setattr(forward, "PARALLEL_MIN_POINTS", parallel_min)

    traj = _traj()
    rng = np.random.default_rng(3)
    obs = OBS + np.c_[rng.uniform(-4000, 4000, (50, 2)), rng.uniform(0, 80, 50)]
    t = np.linspace(-320.0, 520.0, 300)  # the track spans -300 .. 498 s
    fast = log_doppler(traj, obs[:, None, :], t[None, :], 340.29, atmos=atmos)
    ref = log_doppler_numpy(traj, obs[:, None, :], t[None, :], 340.29, atmos=atmos)
    assert fast[0].shape == (50, 300)
    assert (fast[1] == ref[1]).all()
    ok = ref[1]
    assert ok.mean() > 0.6 and (~ok).any()
    assert np.max(np.abs(fast[0] - ref[0])[ok]) < 1e-10
    assert np.max(np.abs(fast[2] - ref[2])[ok]) < 1e-7
