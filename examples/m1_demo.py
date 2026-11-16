"""M1 demo: synthetic flyovers -> noisy frequency tracks -> single-pass fit.

Prints truth vs estimate (+-1 sigma) for a few typical geometries.
Run: .venv/bin/python examples/m1_demo.py
"""

import numpy as np

from dopplergeo.core import SPEED_OF_SOUND_15C as C
from dopplergeo.core import StraightLineFlight, fit_doppler, synthesize_track

CASES = [
    # name,                 v m/s, d m,    f0 Hz, window s,  sigma Hz
    ("approach, low",        75.0,  600.0, 180.0, (-40, 40), 0.3),
    ("departure climb",     110.0, 2500.0, 240.0, (-60, 60), 0.5),
    ("cruise overflight",   230.0, 11000.0, 95.0, (-120, 150), 0.4),
    ("approach only (no CPA)", 80.0, 900.0, 200.0, (-60, -5), 0.3),
]

rng = np.random.default_rng(1)
observer = (0.0, 0.0, 1.5)
header = f"{'case':24} {'param':6} {'truth':>10} {'estimate':>10} {'1 sigma':>9}  grade"
print(header)
print("-" * len(header))
for name, v, d, f0, (a, b), sig in CASES:
    tau0 = 0.0
    flight = StraightLineFlight.from_cpa(v, d, tau0, heading_deg=rng.uniform(0, 360), observer=observer)
    t = np.arange(a, b, 0.1) + d / C
    f = synthesize_track(flight, observer, f0, t, sigma_f=sig, rng=rng)
    fit = fit_doppler(t, f)
    for p, truth in (("f0", f0), ("v", v), ("d", d), ("t_cpa", tau0 + d / C)):
        est = fit.t_cpa if p == "t_cpa" else getattr(fit, p)
        print(f"{name if p == 'f0' else '':24} {p:6} {truth:10.2f} {est:10.2f} {fit.stderr(p):9.2f}  "
              f"{fit.grade if p == 'f0' else ''}")
    print()
