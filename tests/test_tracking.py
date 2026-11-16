"""Spectrogram, peaks, linking and grouping on controlled signals."""

import numpy as np
import pytest
from scipy.ndimage import median_filter

from dopplergeo.core.spectral import (
    SpectroSettings,
    compute_spectrogram,
    running_median_floor,
    subtract_noise,
)
from dopplergeo.core.tracking import (
    FramePeaks,
    ToneTrack,
    group_families,
    link_tracks,
    pick_peaks,
    track_from_guide,
)

FS = 8000


def chirp(f_of_t, duration, amp=1.0, fs=FS):
    t = np.arange(int(duration * fs)) / fs
    phase = 2 * np.pi * np.cumsum(f_of_t(t)) / fs
    return t, amp * np.sin(phase)


def test_fast_floor_matches_median_filter():
    rng = np.random.default_rng(0)
    db = 10 * np.log10(rng.exponential(size=(2000, 30)))
    fast = running_median_floor(db, 61)
    exact = median_filter(db, size=(61, 1), mode="nearest")
    assert np.median(np.abs(fast - exact)) < 0.3  # dB


def test_peak_frequency_accuracy_on_steady_tone():
    _, x = chirp(lambda t: np.full_like(t, 437.3), 10.0)
    x = x + np.random.default_rng(1).normal(0, 0.1, x.size)
    spec = compute_spectrogram(x, FS, SpectroSettings(fmax=2000))
    peaks = [p for p in pick_peaks(spec) if p is not None]
    best = np.array([p.f[np.argmax(p.snr_db)] for p in peaks])
    assert np.median(np.abs(best - 437.3)) < 0.05
    sig = np.array([p.sigma_f[np.argmax(p.snr_db)] for p in peaks])
    assert np.std(best - 437.3) < 3 * np.median(sig)  # reported sigma is not optimistic by >3x


def test_mutes_and_edges_invalidate_frames():
    _, x = chirp(lambda t: np.full_like(t, 300.0), 20.0)
    spec = compute_spectrogram(x, FS, SpectroSettings(fmax=1000), mutes=[(8.0, 10.0)])
    t = spec.times
    assert not spec.frame_valid[(t > 7.6) & (t < 10.4)].any()
    assert spec.frame_valid[(t > 1) & (t < 7)].all()
    assert not spec.frame_valid[t < 0.5].any()


def test_spectral_subtraction_lowers_stationary_hum():
    t, x = chirp(lambda t: np.full_like(t, 200.0), 20.0, amp=0.2)
    hum = 0.5 * np.sin(2 * np.pi * 600 * t)
    tone_late = np.where(t > 10, np.sin(2 * np.pi * 900 * t), 0.0)
    spec = compute_spectrogram(x + hum + tone_late, FS, SpectroSettings(fmax=1500))
    clean = subtract_noise(spec, (1.0, 9.0))
    i600 = np.argmin(np.abs(spec.freqs - 600))
    i900 = np.argmin(np.abs(spec.freqs - 900))
    late = spec.times > 12
    assert clean.db[i600, late].mean() < spec.db[i600, late].mean() - 10
    assert clean.db[i900, late].mean() == pytest.approx(spec.db[i900, late].mean(), abs=1)


def test_linking_keeps_identity_through_a_crossing():
    # Rising and falling tones cross at t = 10 s, 250 Hz.
    _, up = chirp(lambda t: 200 + 5 * t, 20.0)
    _, down = chirp(lambda t: 300 - 5 * t, 20.0, amp=0.7)
    x = up + down + np.random.default_rng(2).normal(0, 0.05, up.size)
    spec = compute_spectrogram(x, FS, SpectroSettings(fmax=1000))
    tracks = link_tracks(pick_peaks(spec), spec.times)
    long = [tr for tr in tracks if tr.duration > 15]
    assert len(long) == 2
    for tr in long:
        slope = np.polyfit(tr.t, tr.f, 1)[0]
        assert abs(abs(slope) - 5) < 0.2  # each stayed on its own chirp


def test_noise_only_gives_no_tracks():
    x = np.random.default_rng(3).normal(size=FS * 30)
    spec = compute_spectrogram(x, FS, SpectroSettings(fmax=3000))
    assert link_tracks(pick_peaks(spec), spec.times) == []


def _track(i, frames, f, k=1.0):
    frames = np.asarray(frames)
    n = frames.size
    return ToneTrack(i, frames, frames * 0.1, np.asarray(f, float), np.full(n, 30.0), np.full(n, 0.03), k=k)


def test_grouping_by_constant_ratio_and_missing_fundamental():
    fr = np.arange(0, 300)
    doppler = 1 + 0.2 * np.tanh((fr - 150) / 40)  # shared sweep
    other = 1 + 0.2 * np.tanh((fr - 200) / 30)  # a different aircraft
    tracks = [
        _track(0, fr, 400 * doppler),  # 2 x f0 (fundamental 200 Hz missing)
        _track(1, fr, 600 * doppler),  # 3 x f0
        _track(2, fr, 450 * other),
    ]
    fams = group_families(tracks, hop_s=0.1)
    assert len(fams) == 2
    fam = next(f for f in fams if len(f.tracks) == 2)
    assert fam.ks == [2.0, 3.0]


def test_flat_tones_are_not_grouped():
    fr = np.arange(0, 200)
    fams = group_families([_track(0, fr, np.full(fr.size, 100.0)), _track(1, fr, np.full(fr.size, 300.0))], 0.1)
    assert len(fams) == 2


def test_pieces_of_one_harmonic_are_merged():
    fr = np.arange(0, 300)
    d = 1 + 0.2 * np.tanh((fr - 150) / 40)
    tracks = [
        _track(0, fr, 200 * d),
        _track(1, fr[:140], 400 * d[:140]),
        _track(2, fr[130:], 400 * d[130:]),
    ]
    fams = group_families(tracks, hop_s=0.1)
    assert len(fams) == 1
    k2 = [tr for tr in fams[0].tracks if tr.k == 2.0]
    assert len(k2) == 1 and k2[0].frame.size == 300


def test_manual_guide_follows_a_weak_tone():
    _, x = chirp(lambda t: 300 - 4 * t, 20.0, amp=0.05)
    x = x + np.random.default_rng(4).normal(0, 0.2, x.size)
    spec = compute_spectrogram(x, FS, SpectroSettings(fmax=1000))
    tr = track_from_guide(spec, np.array([0.0, 20.0]), np.array([301.0, 219.0]), half_width_hz=6)
    assert tr.manual
    assert np.median(np.abs(tr.f - (300 - 4 * tr.t))) < 0.3


def test_link_tracks_skips_invalid_frames():
    peaks = [None, FramePeaks(np.array([100.0]), np.array([20.0]), np.array([0.03]))]
    assert link_tracks(peaks, np.array([0.0, 0.1])) == []  # too short, but no crash
