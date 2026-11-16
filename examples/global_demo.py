"""Worldwide search on real traffic: synthetic tones from real aircraft near a
hidden observer (see m4_demo.py), searched against every aircraft in the world.

    .venv/bin/python examples/global_demo.py --observer 52.2800,4.8100 --max-aircraft 2
"""

from __future__ import annotations

import time

import m4_demo as D
from dopplergeo.core.geo import haversine_m
from dopplergeo.core.solver import SolverSettings
from dopplergeo.flights import BBox, FlightCache, query_window, tracks_from_table
from dopplergeo.flights.ground import adsb_ground_elevation
from dopplergeo.global_search import solve_global


def main() -> None:
    args = D.parser().parse_args()
    sc = D.build(args)
    fams, epoch = sc["fams"], sc["epoch"]
    t = time.perf_counter()
    q = query_window(epoch, args.duration, BBox.world(), clock_bound_s=abs(args.clock_error) + 5)
    with FlightCache(args.cache) as cache:
        table = cache.query(q, "adsblol")
    tracks = tracks_from_table(table)
    ground = adsb_ground_elevation(table)
    print(f"world: {table.num_rows:,} points, {len(tracks):,} airborne tracks, "
          f"{len(ground.site_lat)} airport ground sites  ({time.perf_counter() - t:.1f} s)")
    s = SolverSettings(clock_bound_s=abs(args.clock_error) + 5 if args.clock_error else 0.0)
    t = time.perf_counter()
    res = solve_global(fams, tracks, epoch, s, ground=ground, log=lambda m: print("  " + m))
    print(f"search {time.perf_counter() - t:.1f} s; ambiguous {res.ambiguous}; notes {res.notes}")
    for i, p in enumerate(res.places()[:6]):
        err = haversine_m(p["lat"], p["lon"], *sc["truth"])
        print(f"  #{i + 1} {p['lat']:.5f}, {p['lon']:.5f} {p['kind']:8} chi2_red {p['chi2_red']:8.2f} "
              f"fam {p['families']} {p['aircraft']}  -> {err / 1000:9.3f} km from truth")


if __name__ == "__main__":
    main()
