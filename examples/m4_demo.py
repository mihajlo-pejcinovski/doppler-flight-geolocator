"""M4 demo on real traffic: synthetic tones over real ADS-B tracks.

Takes a cached adsb.lol day, picks real aircraft that pass near a hidden
observer, synthesises their tone curves from the real trajectories (plus
noise), then runs the full solver against *all* aircraft in the area, which
are the decoys gating has to reject.

    .venv/bin/python examples/m4_demo.py --cache ./cache/flights --date 2026-11-12
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from dopplergeo.core.atmosphere import SPEED_OF_SOUND_15C
from dopplergeo.core.doppler import fit_doppler
from dopplergeo.core.geo import geodetic_to_enu, haversine_m
from dopplergeo.core.solver import FamilyObs, Frame, SampledTrajectory, SolverSettings, log_doppler, solve
from dopplergeo.flights import BBox, FlightCache, query_window, tracks_from_table


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/flights")
    ap.add_argument("--date", default="2026-11-12")
    ap.add_argument("--start", default="14:02:00", help="recording start, UTC")
    ap.add_argument("--observer", default="52.2800,4.8100", help="hidden true position lat,lon")
    ap.add_argument("--radius", type=float, default=50.0, help="search radius km")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--max-aircraft", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.05, help="tone noise Hz")
    ap.add_argument("--elevation", default="auto", choices=["auto", "dem", "adsb", "constant"],
                    help="observer height source given to the solver")
    ap.add_argument("--clock-error", type=float, default=0.0, help="file clock error s (true = file + this)")
    return ap



def build(args: argparse.Namespace, verbose: bool = True) -> dict:
    """Real tracks + hidden observer -> synthetic families. Returns the scenario."""
    lat_o, lon_o = (float(x) for x in args.observer.split(","))
    epoch = datetime.fromisoformat(f"{args.date}T{args.start}+00:00").timestamp()
    region = BBox.around(lat_o + 0.07, lon_o - 0.05, args.radius)  # centre deliberately off the truth
    q = query_window(epoch, args.duration, region, clock_bound_s=abs(args.clock_error) + 5)
    with FlightCache(args.cache) as cache:
        table = cache.query(q, "adsblol")
        tracks = tracks_from_table(table)
    # The hidden observer stands on the real terrain (DEM); the solver gets the chosen source.
    from dopplergeo.core.elevation import copernicus_dem
    from dopplergeo.flights.ground import observer_elevation
    cache_root = str(__import__("pathlib").Path(args.cache).parent)
    terrain = copernicus_dem(region.expanded(10).as_tuple(), cache_dir=cache_root)
    h_true = float(terrain.height(lat_o, lon_o)[0])
    elev, notes = observer_elevation(args.elevation, region, table, cache_dir=cache_root)
    if verbose:
        print(f"{len(tracks)} airborne tracks in the window")

    frame = Frame(*region.center, epoch)
    obs = geodetic_to_enu(lat_o, lon_o, h_true, *region.center)
    rng = np.random.default_rng(3)
    fams, used = [], []
    for tr in sorted(tracks, key=lambda tr: haversine_m(tr.lat, tr.lon, lat_o, lon_o).min()):
        if len(fams) >= args.max_aircraft:
            break
        st = SampledTrajectory(tr, *region.center, epoch)
        t_true = np.arange(5.0, args.duration - 5.0, 0.1)
        logD, valid, tau = log_doppler(st, obs, t_true, SPEED_OF_SOUND_15C)  # same c as the solver
        P, V, _ = st.eval(tau)
        slant = np.linalg.norm(P - obs, axis=-1)
        speed = np.hypot(V[:, 0], V[:, 1])
        audible = valid & (slant < 8000) & (speed > 50)
        if audible.sum() < 300:
            continue
        # Keep one contiguous audible stretch (a single pass).
        idx = np.flatnonzero(audible)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        if len(run) < 300:
            continue
        f0 = float(rng.uniform(70, 160))
        t, f = t_true[run], f0 * np.exp(logD[run]) + rng.normal(0, args.noise, len(run))
        sig = np.full(t.size, max(args.noise, 0.01))
        try:
            fit = fit_doppler(t - args.clock_error, f, sigma=sig)
        except ValueError:
            continue
        fams.append(FamilyObs(len(fams), t - args.clock_error, f, np.ones(t.size), sig, fit))
        used.append(tr)
        if verbose:
            print(f"  family {len(fams) - 1}: {tr.icao24} {tr.callsign or '-':8} min slant {slant[run].min():5.0f} m"
                  f"  fit v={fit.v:5.1f} d={fit.d:5.0f} grade {fit.grade}")

    return {"fams": fams, "used": used, "tracks": tracks, "epoch": epoch, "region": region,
            "truth": (lat_o, lon_o), "h_true": h_true, "elevation": elev}


def main() -> None:
    args = parser().parse_args()
    sc = build(args)
    fams, used, tracks, epoch, region = sc["fams"], sc["used"], sc["tracks"], sc["epoch"], sc["region"]
    lat_o, lon_o = sc["truth"]
    s = SolverSettings(clock_bound_s=abs(args.clock_error) + 5 if args.clock_error else 0.0,
                       elevation=sc["elevation"])
    t = time.perf_counter()
    sol = solve(fams, tracks, epoch, region, s)
    dt = time.perf_counter() - t
    n_c = len(sol.candidates)
    a, b, ang = sol.ellipse_95
    print(f"\nsolve {dt:.1f} s, grid {sol.grid.chi2.size:,} cells, {n_c} candidates from {len(tracks)} aircraft")
    for c in sorted(sol.candidates, key=lambda c: (c.family_id, c.chi2)):
        print(f"  fam {c.family_id} {c.icao24} {c.callsign or '-':8} speed x{c.speed_match:4.2f} chi2 {c.chi2:12.1f}")
    print(f"truth     {lat_o:.5f}, {lon_o:.5f}")
    print(f"solution  {sol.lat:.5f}, {sol.lon:.5f}  error {haversine_m(sol.lat, sol.lon, lat_o, lon_o):.1f} m")
    print(f"ellipse   {a:.0f} x {b:.0f} m @ {ang:.0f} deg   dt {sol.dt_clock:+.3f} s   chi2_red {sol.chi2_red:.2f}")
    print(f"height    {sol.h:.1f} m (true {sc['h_true']:.1f}; {sol.h_source} prior {sol.h_prior:.1f}, ±{sol.h_sigma:.1f} {'posterior' if sol.h_fitted else 'prior'}"
          f")")
    print(f"grade     {sol.grade}  resolved {sol.resolved}  aircraft {sol.aircraft_used}  "
          f"assignments correct {all(used[i].icao24 == ic for i, ic in sol.assignment.items())}  "
          f"unexplained {sol.unexplained}")
    if sol.mirror:
        print(f"mirror    {sol.mirror['lat']:.5f}, {sol.mirror['lon']:.5f}  delta chi2 {sol.mirror['delta_chi2']:.1f}")


if __name__ == "__main__":
    main()
