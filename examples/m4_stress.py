"""Many random M4 scenarios on real traffic (see m4_demo.py): accuracy vs grade.

    .venv/bin/python examples/m4_stress.py --n 12 --seed 1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import m4_demo as D
from dopplergeo.core.geo import haversine_m
from dopplergeo.core.solver import SolverError, SolverSettings, solve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--cache", default="./cache/flights")
    ap.add_argument("--date", default="2026-11-12")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(a.n):
        lat = 52.31 + rng.uniform(-0.12, 0.12)
        lon = 4.76 + rng.uniform(-0.2, 0.2)
        hh, mm = rng.integers(7, 20), rng.integers(0, 60)
        clock = float(rng.choice([0.0, 0.0, rng.uniform(-30, 30)]))
        noise = float(rng.choice([0.03, 0.05, 0.15]))
        args = D.parser().parse_args([
            "--cache", a.cache, "--date", a.date, "--start", f"{hh:02d}:{mm:02d}:00",
            "--observer", f"{lat:.4f},{lon:.4f}", "--radius", str(a.radius),
            "--noise", str(noise), "--clock-error", f"{clock:.2f}"])
        sc = D.build(args, verbose=False)
        if not sc["fams"]:
            print(f"{i:2d} no audible aircraft, skipped")
            continue
        s = SolverSettings(clock_bound_s=abs(clock) + 5 if clock else 0.0, elevation=sc["elevation"])
        t = time.perf_counter()
        try:
            sol = solve(sc["fams"], sc["tracks"], sc["epoch"], sc["region"], s)
        except SolverError as e:
            print(f"{i:2d} {len(sc['fams'])} fam  no solution: {e}")
            rows.append(("-", float("inf"), time.perf_counter() - t))
            continue
        dt = time.perf_counter() - t
        err = float(haversine_m(sol.lat, sol.lon, *sc["truth"]))
        ok_asg = all(sc["used"][k].icao24 == v for k, v in sol.assignment.items())
        a95 = sol.ellipse_95[0]
        rows.append((sol.grade, err, dt))
        print(f"{i:2d} {len(sc['fams'])} fam clock {clock:+6.1f} noise {noise:.2f}  err {err:9.1f} m  "
              f"95% {a95:8.1f} m  grade {sol.grade} chi2r {sol.chi2_red:8.2f}  asg ok {ok_asg!s:5}  "
              f"h {sol.h - sc['h_true']:+5.1f}  unexpl {sol.unexplained}  {dt:5.1f} s  mirror {'yes' if sol.mirror else 'no'}")
    print()
    for g in "ABC-":
        errs = [e for gg, e, _ in rows if gg == g]
        if errs:
            print(f"grade {g}: {len(errs):2d}  median err {np.median(errs):9.1f} m  max {max(errs):9.1f} m")
    print(f"solve time median {np.median([r[2] for r in rows]):.1f} s, max {max(r[2] for r in rows):.1f} s")


if __name__ == "__main__":
    main()
