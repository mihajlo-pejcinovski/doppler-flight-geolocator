"""Manchester: chi2 profile along the ellipse's major axis, per air model.

    .venv/bin/python examples/manchester_profile.py cache/analysis/manchester_full.pkl

For each model the fair solve runs first (as in manchester_timing.py), then the
observer is held at points along the 95% ellipse's major axis (across the
approach) and everything else (clock, lags, height, rest frequencies) is
refitted. The profile is printed in the covariance's own units (``chi2_info``:
a rise of 1 = one sigma squared) next to what the ellipse claims, (o / sigma)^2.
A profile much flatter than the claim means the ellipse is overconfident; a
non-parabolic or two-dipped one means the minimum is not well defined.

Models: still (constant 15 C, the old model), lapse (15 C, ISA lapse, calm). (A
"flat" moving-air model with no lapse and no wind reproduced still exactly.)
"""

from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dopplergeo.core.atmosphere import Atmosphere
from dopplergeo.core.elevation import ConstantElevation
from dopplergeo.core.geo import geodetic_to_enu, haversine_m
from dopplergeo.core.solver import SolverError, SolverSettings, refine
from dopplergeo.flights import FlightCache
from dopplergeo.flights.adsblol import AdsbLolProvider
from dopplergeo.locate import family_observations, fetch_flights, resolve_region, solve_position

ROOT = Path(__file__).resolve().parents[1]
GEOTAG = (53.44054, -2.099867)
EPOCH = datetime(2026, 9, 3, 11, 0, tzinfo=timezone.utc).timestamp()
BOUND = 600.0
MODELS = {
    "still": None,
    "lapse": Atmosphere(temp_c=15.0, lapse=True, source="lapse"),
}
OFFSETS = np.arange(-800.0, 401.0, 100.0)
_CTX: dict = {}


def run(name: str) -> str:
    atm = MODELS[name]
    res, region, region_src, data, cache, fams = (_CTX[k] for k in ("res", "region", "region_src", "data",
                                                                     "cache", "fams"))
    base = SolverSettings(adsb_lag_sigma_s=0.5, atmos=atm)
    r = solve_position(res, EPOCH, "user", region, region_src, data, clock_bound_s=BOUND, cache=cache,
                       clock_sigma_s=0.0, weather="none", settings=base)
    sol = r.solution
    if sol is None:
        return f"{name}: NO SOLUTION ({r.error})"
    best, frame = sol.best, sol.frame
    err = float(haversine_m(sol.lat, sol.lon, *GEOTAG))
    a, b, ang = sol.ellipse_95
    sig = a / math.sqrt(5.991)  # 1-sigma along the major axis
    u = np.array([math.sin(math.radians(ang)), math.cos(math.radians(ang))])
    g = np.array(geodetic_to_enu(*GEOTAG, sol.h, frame.lat0, frame.lon0)[:2]) - np.array([best.e, best.n])
    s = SolverSettings(adsb_lag_sigma_s=0.5, clock_bound_s=BOUND, atmos=atm,
                       c=atm.c_surface if atm is not None else SolverSettings.c,
                       elevation=ConstantElevation(best.h_prior, best.h_sigma if best.h_fitted else 5.0, "profile"))
    used = [f for f in fams if f.id in best.assignment]
    lines = [f"== {name}: {err:.0f} m from geotag, ellipse {a:.0f} x {b:.0f} m @ {ang:.0f} deg "
             f"(1 sigma along major {sig:.0f} m); geotag at {g @ u:+.0f} m along that axis, {g @ [u[1], -u[0]]:+.0f} m "
             f"off it; clock {sol.dt_clock:+.2f} s"]
    sig_minor = b / math.sqrt(5.991)
    w = np.array([u[1], -u[0]])  # minor axis (along the approach for Manchester)
    for label, axis, sg, geo in (("major axis (across track)", u, sig, g @ u),
                                 ("minor axis (along track)", w, sig_minor, g @ w)):
        lines.append(f"  -- {label}: 1 sigma claimed {sg:.0f} m, geotag at {geo:+.0f} m")
        rows = []
        for o in OFFSETS:
            e, n = best.e + o * axis[0], best.n + o * axis[1]
            try:
                fit = refine(used, best.assignment, frame, e, n, best.dt, s, "profile", hold_position=True)
            except (SolverError, ValueError, np.linalg.LinAlgError) as ex:
                rows.append((o, float("nan"), float("nan"), str(ex)))
                continue
            rows.append((o, fit.chi2_eff, fit.chi2_scale, f"clock {fit.dt:+.2f}  chi2_red {fit.chi2_red:.3f}"))
        finite = [(c, sc) for _, c, sc, _ in rows if np.isfinite(c)]
        c0, sc0 = min(finite) if finite else (float("nan"), float("nan"))
        for o, c, _, note in rows:
            lines.append(f"  {o:+6.0f} m   data rise {(c - c0) / sc0:8.1f}   ellipse claims {(o / sg) ** 2:8.1f}   {note}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    res = pickle.load(open(argv[0], "rb"))
    names = argv[1:] or list(MODELS)
    region, region_src = resolve_region((53.3537, -2.2750), 25.0, None)
    fams = family_observations(res)
    cache = FlightCache(ROOT / "cache/flights")
    with cache:
        data = fetch_flights(fams, EPOCH, region, clock_bound_s=BOUND,
                             providers=[AdsbLolProvider(confirm=lambda *a: False)], cache=cache)
    _CTX.update(res=res, region=region, region_src=region_src, data=data, cache=cache, fams=fams)
    with mp.get_context("fork").Pool(len(names)) as pool:
        for out in pool.imap(run, names):
            print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
