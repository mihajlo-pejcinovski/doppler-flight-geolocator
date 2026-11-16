"""End to end: rendered flyovers -> (AAC file) -> analyze -> families and fits."""

import csv

import numpy as np
import pytest

from dopplergeo.cli import main
from dopplergeo.core import SPEED_OF_SOUND_15C, StraightLineFlight, render_audio
from dopplergeo.core.media import write_audio_file
from dopplergeo.core.pipeline import analyze, analyze_file
from dopplergeo.core.synthetic import add_wind_noise

C = SPEED_OF_SOUND_15C
OBS = (0.0, 0.0, 1.5)
FS = 16_000


def mix(flights, duration=100.0, snr_db=10.0, wind_db=None, seed=0):
    x = 0.0
    for flight, tones in flights:
        _, xi = render_audio(flight, OBS, tones, fs=FS, duration=duration, snr_db=None)
        x = x + xi
    rng = np.random.default_rng(seed)
    x = x + rng.normal(0, np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)), x.size)
    if wind_db is not None:
        x = add_wind_noise(x, FS, wind_db, rng=seed + 1)
    return x


def assert_fit(fr, v, d, tau0, f0=None):
    f = fr.fit
    assert f is not None and f.grade in ("A", "B")
    assert f.v == pytest.approx(v, rel=0.02)
    assert f.d == pytest.approx(d, rel=0.05)
    assert f.t_cpa == pytest.approx(tau0 + d / C, abs=0.5)
    if f0 is not None:
        assert f.f0 == pytest.approx(f0, rel=0.002)


def test_single_flyover_with_wind_through_aac(tmp_path):
    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 50.0, 70.0, observer=OBS)
    x = mix([(flight, [(220, 1.0), (440, 0.5), (660, 0.3)])], wind_db=0.0)
    path = tmp_path / "flyover.m4a"
    write_audio_file(path, x, FS, bit_rate=96_000, metadata={"creation_time": "2026-11-13T10:00:00Z"})
    res = analyze_file(path)
    assert len(res.reliable) == 1
    fr = res.reliable[0]
    assert fr.family.ks == [1.0, 2.0, 3.0]
    assert_fit(fr, 90.0, 1500.0, 50.0, f0=220.0)
    assert fr.flags["cpa_covered"] and fr.flags["asymptotes_visible"]


def test_two_aircraft_with_crossing_tones_stay_separate():
    a = StraightLineFlight.from_cpa(90.0, 1500.0, 40.0, 70.0, observer=OBS)
    b = StraightLineFlight.from_cpa(120.0, 900.0, 60.0, 200.0, observer=OBS)
    res = analyze(mix([(a, [(220, 1.0), (440, 0.5)]), (b, [(240, 1.0), (480, 0.6)])]), FS)
    assert len(res.reliable) == 2
    by_v = sorted(res.reliable, key=lambda fr: fr.fit.v)
    assert_fit(by_v[0], 90.0, 1500.0, 40.0, f0=220.0)
    assert_fit(by_v[1], 120.0, 900.0, 60.0, f0=240.0)


def test_fast_jet_with_missing_fundamental():
    jet = StraightLineFlight.from_cpa(160.0, 2200.0, 40.0, 10.0, observer=OBS)
    res = analyze(mix([(jet, [(440, 0.6), (660, 0.4)])]), FS)
    assert len(res.reliable) == 1
    fr = res.reliable[0]
    assert fr.family.ks == [2.0, 3.0]
    assert_fit(fr, 160.0, 2200.0, 40.0, f0=220.0)


def test_muted_speech_burst_is_ignored():
    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 50.0, 70.0, observer=OBS)
    x = mix([(flight, [(220, 1.0), (440, 0.5)])])
    t = np.arange(x.size) / FS
    burst = (t > 20) & (t < 25)
    x[burst] += 3.0 * np.sin(2 * np.pi * 330 * t[burst])  # loud steady interferer
    res = analyze(x, FS, mutes=[(20.0, 25.0)])
    assert len(res.reliable) == 1
    assert_fit(res.reliable[0], 90.0, 1500.0, 50.0)


def test_cli_writes_csv(tmp_path, capsys):
    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 50.0, 70.0, observer=OBS)
    path = tmp_path / "clip.m4a"
    write_audio_file(path, mix([(flight, [(220, 1.0), (440, 0.5)])], duration=60.0), FS,
                     metadata={"creation_time": "2026-11-13T10:00:00Z", "location": "+48.85+002.35/"})
    out = tmp_path / "out"
    assert main(["analyze", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "2026-11-13T10:00:00+00:00 (creation_time)" in text
    assert "(48.85, 2.35, None)" in text
    with (out / "fits.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["grade"] in ("A", "B")
    assert rows[0]["t_cpa_utc"].startswith("2026-11-13T10:00:5")
    assert float(rows[0]["v_mps"]) == pytest.approx(90.0, rel=0.02)
    with (out / "tracks.csv").open() as fh:
        pts = list(csv.DictReader(fh))
    assert len(pts) > 500 and {r["k"] for r in pts} == {"1", "2"}


def test_cli_start_utc_override(tmp_path, capsys):
    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 20.0, 70.0, observer=OBS)
    path = tmp_path / "clip.wav"
    write_audio_file(path, mix([(flight, [(220, 1.0)])], duration=40.0), FS, codec="pcm_s16le")
    out = tmp_path / "out"
    assert main(["analyze", str(path), "--out", str(out), "--start-utc", "2026-01-01T12:00:00Z"]) == 0
    assert "(user)" in capsys.readouterr().out


def test_windowed_analysis_keeps_each_pass_once():
    """Three passes, windows of 200 s overlapping 100 s (starts 0, 100 s; cores split
    at 150 s). The middle pass (CPA 141 s) straddles the second window's start: it
    must be fitted once, by the first window, whose core holds its CPA with at least
    half the overlap on each side (a pass cut 40 s after its CPA fits d ~5% long)."""
    from dopplergeo.core.pipeline import analyze_windowed

    # One tone each, in separate bands: crossing tones are a tracker matter, not this test's.
    passes = [(90.0, 900.0, 50.0, 30.0, 200.0), (110.0, 800.0, 140.0, 120.0, 900.0),
              (80.0, 700.0, 240.0, 250.0, 400.0)]
    flights = [(StraightLineFlight.from_cpa(v, d, tau, hd, observer=OBS), [(f0, 1.0)])
               for v, d, tau, hd, f0 in passes]
    x = mix(flights, duration=300.0)
    # Hum filter off: in a 200 s window the last pass sits near its high asymptote for
    # ~100 s, which the 150 s steady-line filter removes (a separate, known limit).
    from dopplergeo.core.tracking import TrackerSettings

    res = analyze_windowed(x, FS, window_s=200.0, overlap_s=100.0, tracker=TrackerSettings(stationary_window_s=0))
    good = sorted(res.reliable, key=lambda fr: fr.fit.t_cpa)
    assert len(good) == 3, [(fr.fit.t_cpa, fr.fit.v) for fr in res.families if fr.fit]
    for fr, (v, d, tau, _, f0) in zip(good, passes):
        # Loose d / t_cpa tolerances: these single-tone fits are off by as much
        # unwindowed (d +6% at 200 Hz; t_cpa -1 s at 900 Hz, 110 m/s).
        f = fr.fit
        assert f.v == pytest.approx(v, rel=0.02) and f.d == pytest.approx(d, rel=0.08)
        assert f.t_cpa == pytest.approx(tau + d / C, abs=1.5) and f.f0 == pytest.approx(f0, rel=0.002)
    # The middle pass's tail in the last window must defer to the window holding its CPA.
    cpas = sorted(fr.fit.t_cpa for fr in res.families if fr.fit is not None and fr.fit.success)
    assert all(b - a > 5.0 for a, b in zip(cpas, cpas[1:])), cpas
    assert len({fr.family.id for fr in res.families}) == len(res.families)
    assert len({tr.id for tr in res.tracks}) == len(res.tracks)


def test_windowed_overview_spectrogram_spans_the_recording(monkeypatch):
    """The display spectrogram of a windowed analysis is every window's core,
    pooled, in time order, without gaps or repeats at the joins."""
    from dopplergeo.core import pipeline as P

    monkeypatch.setattr(P, "DISPLAY_MAX_FRAMES", 1000)  # 300 s at 0.1 s hop -> pool 3 frames
    monkeypatch.setattr(P, "DISPLAY_MAX_BINS", 2000)
    flight = StraightLineFlight.from_cpa(90.0, 900.0, 150.0, 30.0, observer=OBS)
    x = mix([(flight, [(300.0, 1.0)])], duration=300.0)
    res = P.analyze_windowed(x, FS, window_s=200.0, overlap_s=100.0)
    spec = res.spectrogram
    dt = np.diff(spec.times)
    assert spec.times[0] < 1.0 and spec.times[-1] > 299.0
    # 0.3 s pooled steps; at each of the 1 joins a core's leftover frames (fewer than
    # a pooling block) may lengthen one step, by less than one block
    assert dt.min() > 0.29 and dt.max() < 0.6, (dt.min(), dt.max())
    assert np.sum(~np.isclose(dt, 0.3, atol=0.02)) <= 1
    assert spec.times.size <= 1010 and spec.freqs.size <= 2000
    assert spec.power.shape == (spec.freqs.size, spec.times.size) == spec.whitened_db().shape
    # the tone is visible in the pooled image at its CPA
    j = int(np.argmin(np.abs(spec.times - (150.0 + 900.0 / C))))
    assert abs(spec.freqs[np.argmax(spec.db[:, j])] - 300.0) < 6.0


def test_analysis_cache_skips_tracking_on_reopen(tmp_path, monkeypatch):
    from dopplergeo.core import pipeline as P

    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 50.0, 70.0, observer=OBS)
    path = tmp_path / "flyover.m4a"
    write_audio_file(path, mix([(flight, [(220, 1.0), (440, 0.5)])]), FS, bit_rate=96_000)
    sha = "ab" * 32
    first, hit1 = P.analyze_file_cached(path, tmp_path / "analysis", sha)
    assert not hit1 and list((tmp_path / "analysis").glob("*.pkl"))

    def no_tracking(*a, **k):
        raise AssertionError("tracking ran although the analysis was cached")
    monkeypatch.setattr(P, "analyze_windowed", no_tracking)
    again, hit2 = P.analyze_file_cached(path, tmp_path / "analysis", sha)
    assert hit2
    assert [fr.fit.v for fr in again.families if fr.fit] == [fr.fit.v for fr in first.families if fr.fit]
    assert again.clip is not None and again.spectrogram.power.shape == first.spectrogram.power.shape
    np.testing.assert_allclose(again.spectrogram.power, first.spectrogram.power)
    # different mutes (or settings, or code) -> a different key -> analysed again
    assert P.analysis_cache_key(sha, mutes=[(1.0, 2.0)]) != P.analysis_cache_key(sha, mutes=[])


def test_locked_speed_is_held_and_refit_does_not_trim():
    from dopplergeo.core.doppler import fit_doppler
    from dopplergeo.core.pipeline import refit_families

    flight = StraightLineFlight.from_cpa(90.0, 1500.0, 50.0, 70.0, observer=OBS)
    res = analyze(mix([(flight, [(220, 1.0), (440, 0.5)])]), FS)
    fr = res.reliable[0]
    t, f, k, s = fr.family.stacked()
    locked = fit_doppler(t, f, k=k, sigma=s, fixed={"v": 95.0})
    assert locked.v == pytest.approx(95.0, rel=1e-5)
    assert not locked.at_bound                     # a lock is not a fit pinned at a bound
    out = refit_families(res.tracks, locks={fr.family.id: {"v": 95.0}})
    got = next(x for x in out if x.family.id == fr.family.id)
    assert got.fit.v == pytest.approx(95.0, rel=1e-5)
    assert sum(tr.t.size for tr in got.family.tracks) == sum(tr.t.size for tr in fr.family.tracks)
