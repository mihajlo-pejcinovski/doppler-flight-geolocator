"""M3 acceptance: time bbox/window queries against an indexed adsb.lol day.

    .venv/bin/python examples/m3_query_bench.py --cache ./cache/flights --date 2026-11-12
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from dopplergeo.flights import BBox, FlightCache, query_window, tracks_from_table

SITES = {
    "Amsterdam Schiphol": (52.31, 4.76),
    "London Heathrow": (51.47, -0.45),
    "Frankfurt": (50.03, 8.57),
    "New York JFK": (40.64, -73.78),
    "Atlanta": (33.64, -84.43),
    "Dubai": (25.25, 55.36),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/flights")
    ap.add_argument("--date", required=True)
    ap.add_argument("--hour", type=int, default=14)
    ap.add_argument("--radius", type=float, default=50.0)
    args = ap.parse_args()
    start = datetime.fromisoformat(args.date).replace(hour=args.hour, tzinfo=timezone.utc).timestamp()
    with FlightCache(args.cache) as cache:
        rec = [r for r in cache.records("adsblol")]
        print(f"parts: {[(r['path'].rsplit('/', 1)[-1], r['n_points']) for r in rec]}")
        print(f"{'site':20} {'query s':>8} {'points':>7} {'tracks':>6} {'spline s':>8}")
        worst = 0.0
        for name, (lat, lon) in SITES.items():
            q = query_window(start, 300.0, BBox.around(lat, lon, args.radius))
            t = time.perf_counter()
            tbl = cache.query(q, "adsblol")
            dq = time.perf_counter() - t
            t = time.perf_counter()
            tracks = tracks_from_table(tbl)
            ds = time.perf_counter() - t
            worst = max(worst, dq)
            print(f"{name:20} {dq:8.3f} {tbl.num_rows:7d} {len(tracks):6d} {ds:8.3f}")
        print(f"worst query {worst:.3f} s -> {'PASS' if worst < 1.0 else 'FAIL'} (< 1 s)")


if __name__ == "__main__":
    main()
