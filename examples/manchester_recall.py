"""Tracker recall on the Manchester recording (Freesound 868926) against ADS-B.

    .venv/bin/python examples/manchester_recall.py [analysis.pkl] [--save analysis.pkl]

Runs the windowed analysis on the 75-min file (or loads a pickled
``AnalysisResult``) and scores the fitted families against the 29 aircraft that
ADS-B shows within 3 km of the recordist (data/freesound/manchester_passes.json):

- found:   a family with a fit whose CPA is within ``T_TOL`` s of the pass
- right:   found, and v within 15 %, d within 35 % of the ADS-B values
- good:    right, and graded A or B (what the solver uses by default)
- junk B:  families graded A/B that match no pass (false confidence)
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "data/freesound/868926_5287430-hq.mp3"
PASSES = ROOT / "data/freesound/manchester_passes.json"
SHIFT = 7.6     # matched passes are heard this much after the ADS-B prediction (clock / geotag)
T_TOL = 15.0


def score(res, passes):
    fits = [fr for fr in res.families if fr.fit is not None and fr.fit.success]
    used = set()
    rows = []
    for p in passes:
        t_true = p["t_cpa_media_s"] + SHIFT
        cands = [fr for fr in fits if abs(fr.fit.t_cpa - t_true) <= T_TOL]
        right = [fr for fr in cands if abs(fr.fit.v - p["v_mps"]) <= 0.15 * p["v_mps"]
                 and abs(fr.fit.d - p["d_m"]) <= 0.35 * p["d_m"]]
        best = min(right or cands, key=lambda fr: ("ABC".index(fr.fit.grade), abs(fr.fit.t_cpa - t_true)),
                   default=None)
        if best is not None:
            used.add(id(best))
        rows.append((p, best, bool(right), bool(right) and best.fit.grade in "AB"))
    junk_b = [fr for fr in fits if fr.fit.grade in "AB" and id(fr) not in used]
    return rows, junk_b


def main(argv: list[str]) -> int:
    passes = json.loads(PASSES.read_text())["passes"]
    save = None
    if "--save" in argv:
        i = argv.index("--save")
        save = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if argv:
        res = pickle.load(open(argv[0], "rb"))
        took = None
    else:
        from dopplergeo.core.pipeline import analyze_file
        t = time.time()
        res = analyze_file(MEDIA)
        took = time.time() - t
        if save:
            res.spectrogram = None
            res.clip = None
            pickle.dump(res, open(save, "wb"))
    rows, junk_b = score(res, passes)
    for p, fr, right, good in rows:
        f = fr.fit if fr else None
        got = (f"#{fr.family.id:<3d} t={f.t_cpa:7.1f} v={f.v:5.1f} d={f.d:6.0f} {f.grade}"
               if f else "-- missed --")
        mark = "GOOD" if good else "right" if right else "found" if fr else ""
        print(f"{p['callsign']:9s} t={p['t_cpa_media_s'] + SHIFT:7.1f} v={p['v_mps']:5.1f} d={p['d_m']:5d} | {got:42s} {mark}")
    n = len(rows)
    found = sum(1 for r in rows if r[1] is not None)
    right = sum(1 for r in rows if r[2])
    good = sum(1 for r in rows if r[3])
    print(f"\npasses {n}: found {found}, right {right}, good {good}; junk A/B families {len(junk_b)}; "
          f"families {len(res.families)}" + (f"; analysis {took:.0f} s" if took else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
