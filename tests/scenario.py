"""Shared synthetic scenario: two aircraft passing an observer near Schiphol,
rendered to an AAC file with a creation time and GPS tag, plus a flight cache
holding their ADS-B tracks. Used by the locate and GUI tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dopplergeo.core import StraightLineFlight, render_audio
from dopplergeo.core.geo import enu_to_geodetic, geodetic_to_enu
from dopplergeo.core.media import write_audio_file
from dopplergeo.core.synthetic import adsb_samples
from dopplergeo.flights import BBox, Coverage, FlightCache, TrackPoint, points_to_table

LAT0, LON0 = 52.30, 4.76
START = datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc)
EPOCH = START.timestamp()
H_OBS = 1.5
OBS_LAT, OBS_LON, _ = enu_to_geodetic([2100.0, -1300.0, 0.0], LAT0, LON0)
OBS = geodetic_to_enu(OBS_LAT, OBS_LON, H_OBS, LAT0, LON0)
FS = 16_000


def make_recording(tmp: Path) -> tuple[Path, Path, Path]:
    a = StraightLineFlight.from_cpa(95.0, 1400.0, 45.0, 70.0, 1000.0, observer=OBS)
    b = StraightLineFlight.from_cpa(120.0, 1700.0, 100.0, 200.0, 1300.0, observer=OBS)
    x = 0.0
    for fl, tones in ((a, [(220, 1.0), (440, 0.5)]), (b, [(300, 1.0), (600, 0.5)])):
        _, xi = render_audio(fl, OBS, tones, fs=FS, duration=150.0, snr_db=None)
        x = x + xi
    rng = np.random.default_rng(0)
    x = x + rng.normal(0, np.sqrt(np.mean(x**2) / 10), x.size)  # 10 dB SNR
    media = tmp / "flyover.m4a"
    write_audio_file(media, x, FS, bit_rate=96_000, metadata={
        "creation_time": START.isoformat().replace("+00:00", "Z"),
        "location": f"{OBS_LAT:+.6f}{OBS_LON:+.6f}/"})

    rows = []
    for icao, fl in (("4840d6", a), ("3c6444", b)):
        s = adsb_samples(fl.position, fl.velocity, np.arange(-300.0, 450.0, 2.0), LAT0, LON0)
        for i in range(s["t"].size):
            rows.append(TrackPoint(EPOCH + s["t"][i], icao, icao.upper(), s["lat"][i], s["lon"][i],
                                   s["alt_geo_m"][i], s["alt_baro_m"][i], s["gs_mps"][i], s["track_deg"][i],
                                   s["vrate_mps"][i]))
    cache_dir = tmp / "cache"
    with FlightCache(cache_dir) as c:
        c.write_part(points_to_table(rows), [Coverage("adsblol", EPOCH - 3600, EPOCH + 3600, BBox.world())])
    return media, cache_dir, tmp
