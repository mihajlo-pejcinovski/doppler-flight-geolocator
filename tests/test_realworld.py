"""Failure modes found on real flyover recordings (Freesound clips, M5), as
synthetic regressions: the M2 Kalman linker fragmented every real tone.

* close, fast pass: the pitch falls tens of Hz/s at the CPA (LAX threshold, d ~ 60 m);
* slow source wander (engine RPM) of ~0.2% with broadband noise: per-peak
  Cramer-Rao errors understate the scatter, residuals are serially correlated
  and the robust loss alone cannot rank local minima;
* fits pinned at a bound must not be graded A/B.
"""

from __future__ import annotations

import numpy as np
import pytest

from dopplergeo.core import SPEED_OF_SOUND_15C, StraightLineFlight, render_audio
from dopplergeo.core.doppler import correlation_factor, fit_doppler
from dopplergeo.core.pipeline import analyze
from dopplergeo.core.spectral import SpectroSettings
from dopplergeo.core.tracking import TrackerSettings

C = SPEED_OF_SOUND_15C
OBS = (0.0, 0.0, 1.5)
FS = 16_000


def render(flight, tones, duration, snr_db, seed=0, **kw):
    _, x = render_audio(flight, OBS, tones, fs=FS, duration=duration, snr_db=snr_db, rng=seed, **kw)
    return x


def best(res):
    good = [fr for fr in res.families if fr.fit is not None and fr.fit.grade in ("A", "B")]
    assert good, [(fr.fit.grade, fr.fit.v, fr.fit.d) for fr in res.families if fr.fit]
    return max(good, key=lambda fr: sum(tr.t.size for tr in fr.family.tracks))


@pytest.mark.parametrize("window,hop", [(1.0, 0.1), (0.5, 0.05)])
def test_close_fast_pass_is_tracked_through_the_cpa(window, hop):
    # Turboprop over a runway threshold: 60 m/s, 55 m slant at the CPA. The
    # 480 Hz tone falls ~70 Hz/s at the CPA.
    fl = StraightLineFlight.from_cpa(60.0, 55.0, 15.0, 80.0, altitude=50.0, observer=OBS)
    x = render(fl, [(240, 1.0), (480, 0.6)], 30.0, snr_db=5.0)
    res = analyze(x, FS, spectro=SpectroSettings(window_s=window, hop_s=hop))
    f = best(res).fit
    assert f.v == pytest.approx(60.0, rel=0.05)
    assert f.d == pytest.approx(55.0, rel=0.25)
    assert f.t_cpa == pytest.approx(15.0 + 55.0 / C, abs=0.3)


def test_wandering_tone_in_noise_is_one_track_and_fits():
    # ATR-like: 1 kHz tone with 0.2% RPM wander, weak against broadband noise.
    fl = StraightLineFlight.from_cpa(45.0, 400.0, 45.0, 60.0, observer=OBS)
    x = render(fl, [(1000, 1.0), (2000, 0.5)], 90.0, snr_db=-5.0, seed=3, wander_rel=0.002)
    res = analyze(x, FS)
    fr = best(res)
    f = fr.fit
    assert f.v == pytest.approx(45.0, rel=0.10)
    assert f.d == pytest.approx(400.0, rel=0.30)
    assert f.t_cpa == pytest.approx(45.0 + 400.0 / C, abs=1.0)
    # The tone is recovered as long tracks, not dozens of fragments.
    assert max(tr.duration for tr in fr.family.tracks) > 60.0
    # The error bars account for the correlated wander: the truth is inside 4 sigma.
    assert abs(f.v - 45.0) < 4 * f.stderr("v") + 0.5


def test_kalman_linker_is_still_available():
    fl = StraightLineFlight.from_cpa(90.0, 1500.0, 40.0, 70.0, observer=OBS)
    x = render(fl, [(220, 1.0), (440, 0.5)], 80.0, snr_db=10.0)
    res = analyze(x, FS, tracker=TrackerSettings(method="kalman"))
    assert best(res).fit.v == pytest.approx(90.0, rel=0.02)


def test_fit_multistart_escapes_a_bad_heuristic_start():
    # Flat-then-falling tone with a stray point late in the clip: the slope
    # heuristic puts the CPA too late; the multi-start must recover it.
    rng = np.random.default_rng(1)
    fl = StraightLineFlight.from_cpa(40.0, 300.0, 48.0, 60.0, observer=OBS)
    from dopplergeo.core.doppler import frequency_from_trajectory

    t = np.arange(3.0, 78.0, 0.1)
    f, _ = frequency_from_trajectory(t, 960.0, fl.position, fl.velocity, np.asarray(OBS), C)
    f = f * (1 + 0.003 * np.sin(2 * np.pi * t / 9.0)) + rng.normal(0, 0.3, t.size)
    f[t > 70] -= 40.0  # the ridge slipped onto something else at the end
    fit = fit_doppler(t, f, sigma=np.full(t.size, 2.0))
    assert fit.v == pytest.approx(40.0, rel=0.1)
    assert fit.t_cpa == pytest.approx(48.0 + 300.0 / C, abs=1.0)


def test_fit_pinned_at_a_bound_is_grade_c():
    t = np.linspace(0, 30, 300)
    f = 500.0 - 0.5 * t  # a slow linear slide: no Doppler curve at all
    fit = fit_doppler(t, f + np.random.default_rng(0).normal(0, 0.05, t.size))
    assert fit.at_bound and fit.grade == "C"


def test_correlation_factor():
    rng = np.random.default_rng(0)
    assert correlation_factor(rng.normal(size=5000)) < 1.2
    smooth = np.convolve(rng.normal(size=5000), np.ones(20) / 20, mode="same")
    assert correlation_factor(smooth) > 10


def test_steady_hum_lines_are_removed_and_the_pass_still_fits():
    # Manchester (Freesound 868926): local machinery adds steady lines that the
    # ridge tracker turned into no-Doppler "families" pinned at v = 30 m/s, and
    # a sweeping harmonic was captured by the line it crossed.
    fl = StraightLineFlight.from_cpa(85.0, 800.0, 90.0, 80.0, observer=OBS)
    x = render(fl, [(150, 1.0), (300, 0.7), (450, 0.5)], 180.0, snr_db=5.0, seed=5)
    t = np.arange(x.size) / FS
    rms = np.sqrt(np.mean(x**2))
    for f_hum in (130.0, 330.0, 392.0):  # one sits right on the 300 Hz harmonic's path
        x = x + 0.7 * rms * np.sin(2 * np.pi * f_hum * t)
    res = analyze(x, FS)
    for tr in res.tracks:  # no steady line survives as a track
        assert not any(abs(np.median(tr.f) - h) < 2.0 and np.ptp(tr.f) < 0.01 * h for h in (130, 330, 392))
    f = best(res).fit
    assert f.v == pytest.approx(85.0, rel=0.05)
    assert f.d == pytest.approx(800.0, rel=0.25)
    # Without the removal the lines come back as tracks.
    raw = analyze(x, FS, tracker=TrackerSettings(stationary_window_s=0.0, min_sweep_rel=0.0))
    assert any(np.ptp(tr.f) < 2.0 for tr in raw.tracks)
