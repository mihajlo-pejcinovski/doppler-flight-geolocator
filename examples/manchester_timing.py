"""Manchester (Freesound 868926): solver model variants on a saved analysis.

    .venv/bin/python examples/manchester_timing.py cache/analysis/manchester_full.pkl [variant ...]

Runs the fair setup (region 25 km around MAN, clock +-600 s, geotag withheld)
through ``locate.solve_position`` once per variant and reports the error to the
geotag and whether the geotag is inside the 95% ellipse. The flight data is read
once; the variants then run in parallel worker processes (forked, so the DuckDB
cache is never opened twice). Variants (default: all):

    still        old still-air model, no ADS-B lag
    lag          still air, ADS-B lag sigma 0.5 s (2026-11-15 default)
    lag1         still air, ADS-B lag sigma 1.0 s
    air          Open-Meteo temperature (ISA lapse) + 100 m wind, lag 0.5 s
    air-fitwind  as air, wind fitted with the weather as prior (sigma 3 m/s)
    std          15 deg C calm with the ISA lapse only, lag 0.5 s

The clock prior is not tested here: this clip's file time is a title ("12h00"),
not a phone clock, and its true offset is unknown (-5 s or -13 s).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dopplergeo.core.geo import geodetic_to_enu, haversine_m
from dopplergeo.core.solver import SolverSettings
from dopplergeo.flights import FlightCache
from dopplergeo.flights.adsblol import AdsbLolProvider
from dopplergeo.locate import family_observations, fetch_flights, resolve_region, solve_position

ROOT = Path(__file__).resolve().parents[1]
GEOTAG = (53.44054, -2.099867)
EPOCH = datetime(2026, 9, 3, 11, 0, tzinfo=timezone.utc).timestamp()
BOUND = 600.0
VARIANTS = {
    "still": dict(weather="none", lag=0.0),
    "lag": dict(weather="none", lag=0.5),
    "lag1": dict(weather="none", lag=1.0),
    "air": dict(weather="auto", lag=0.5),
    "air-fitwind": dict(weather="auto", lag=0.5, fit_wind=True),
    "std": dict(weather="standard", lag=0.5),
    # wind and temperature fitted, weather as prior (wind sigma 3 m/s, c sigma 1.7 m/s ~ 3 K)
    "air-fitall": dict(weather="auto", lag=0.5, fit_wind=True, fit_c=True, c_sigma=1.7),
    "std-fitc": dict(weather="standard", lag=0.5, fit_c=True, c_sigma=1.7),
}
APPROACH_DEG = 235.0  # Manchester 23R final approach heading
_CTX: dict = {}


def run(name: str) -> str:
    v = VARIANTS[name]
    res, region, region_src, data, cache = (_CTX[k] for k in ("res", "region", "region_src", "data", "cache"))
    t = time.time()
    r = solve_position(res, EPOCH, "user", region, region_src, data, clock_bound_s=BOUND, cache=cache,
                       clock_sigma_s=0.0, weather=v["weather"], fit_wind=v.get("fit_wind", False),
                       settings=SolverSettings(adsb_lag_sigma_s=v["lag"], fit_c=v.get("fit_c", False),
                                               c_sigma_mps=v.get("c_sigma", 0.0)))
    sol = r.solution
    if sol is None:
        return f"{name:12s} NO SOLUTION ({r.error})"
    err = float(haversine_m(sol.lat, sol.lon, *GEOTAG))
    d = np.array(geodetic_to_enu(*GEOTAG, sol.h, sol.frame.lat0, sol.frame.lon0)[:2]) \
        - np.array([sol.best.e, sol.best.n])
    m2 = float(d @ np.linalg.solve(sol.cov_en, d))
    hd = np.radians(APPROACH_DEG)
    along, across = float(-d @ [np.sin(hd), np.cos(hd)]), float(-d @ [np.cos(hd), -np.sin(hd)])
    a, b, ang = sol.ellipse_95
    calls = {c.icao24: c.callsign for c in sol.candidates}
    out = (f"{name:12s} {err:5.0f} m  grade {sol.grade}  ellipse {a:.0f} x {b:.0f} m @ {ang:.0f} deg  "
           f"geotag {'INSIDE' if m2 < 5.99 else 'outside'} (m2 {m2:.1f})  clock {sol.dt_clock:+.1f} s  "
           f"aircraft {len(sol.aircraft_used)}  [{time.time() - t:.0f} s]"
           f"\n    solution - geotag: {along:+.0f} m along the approach, {across:+.0f} m across (right of track)"
           f";  chi2_red {sol.chi2_red:.3f}, excess {sol.chi2_excess:.2f}")
    if sol.ellipse_jacobian is not None:
        ja, jb, jang = sol.ellipse_jacobian
        mm = sol.ellipse_measure
        out += (f"\n    ellipse: {'measured' if mm else 'Jacobian'}; Jacobian was {ja:.0f} x {jb:.0f} m @ {jang:.0f} deg"
                + (f"; {mm.refits} refits{', re-solved from a better probe' if mm.moved else ''}" if mm else ""))
    if sol.best.atmos is not None:
        out += f"\n    air: {sol.best.atmos.describe()}"
    if sol.best.lags:
        out += "\n    lags: " + ", ".join(f"{calls.get(k) or k} {x:+.2f}" for k, x in sorted(sol.best.lags.items()))
    return out


def main(argv: list[str]) -> int:
    res = pickle.load(open(argv[0], "rb"))
    names = argv[1:] or list(VARIANTS)
    region, region_src = resolve_region((53.3537, -2.2750), 25.0, None)
    fams = family_observations(res)
    cache = FlightCache(ROOT / "cache/flights")
    with cache:
        data = fetch_flights(fams, EPOCH, region, clock_bound_s=BOUND,
                             providers=[AdsbLolProvider(confirm=lambda *a: False)], cache=cache)
    _CTX.update(res=res, region=region, region_src=region_src, data=data, cache=cache)
    with mp.get_context("fork").Pool(min(len(names), 4)) as pool:
        for line in pool.imap(run, names):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
