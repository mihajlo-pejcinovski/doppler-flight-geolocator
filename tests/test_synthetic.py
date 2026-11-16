"""End-to-end on rendered audio: the pitch heard matches the model, and a fit on a
simple spectrogram ridge recovers the flyover geometry."""

import numpy as np
import pytest
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from dopplergeo.core import (
    SPEED_OF_SOUND_15C,
    StraightLineFlight,
    doppler_frequency,
    fit_doppler,
    render_audio,
)

C = SPEED_OF_SOUND_15C
OBS = (0.0, 0.0, 1.5)


def ridge(t, x, fs, f_lo, f_hi, win_s=1.0, hop_s=0.1, min_snr_db=8.0):
    """Peak frequency per frame inside [f_lo, f_hi], with quadratic interpolation.

    Frames whose peak is less than ``min_snr_db`` above the band median are
    dropped (the spec's M2 SNR gate); far from CPA the tone sinks into noise.
    """
    nwin = int(win_s * fs)
    stft = ShortTimeFFT(hann(nwin, sym=False), hop=int(hop_s * fs), fs=fs, mfft=2 * nwin)
    S = np.abs(stft.stft(x)) ** 2
    freqs = stft.f
    times = stft.t(x.size) + t[0]
    keep = (times >= t[0] + win_s / 2) & (times <= t[-1] - win_s / 2)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.nonzero(band)[0]
    sub = np.log(S[band][:, keep] + 1e-30)
    i = np.argmax(sub, axis=0).clip(1, sub.shape[0] - 2)
    cols = np.arange(sub.shape[1])
    a, b, c = sub[i - 1, cols], sub[i, cols], sub[i + 1, cols]
    delta = 0.5 * (a - c) / (a - 2 * b + c)
    df = freqs[1] - freqs[0]
    snr_db = 10 * (b - np.median(sub, axis=0)) / np.log(10)
    good = snr_db >= min_snr_db
    return times[keep][good], (freqs[idx[i]] + delta * df)[good]


@pytest.fixture(scope="module")
def flyover():
    fs, f0 = 8000.0, 220.0
    v, d, tau0 = 90.0, 1500.0, 50.0  # approach speed; k=1 sweeps 299 -> 174 Hz
    flight = StraightLineFlight.from_cpa(v, d, tau0, heading_deg=70.0, observer=OBS)
    t, x = render_audio(
        flight, OBS, [(f0, 1.0), (2 * f0, 0.5)], fs=fs, duration=100.0, snr_db=10.0, rng=0
    )
    return dict(fs=fs, f0=f0, v=v, d=d, tau0=tau0, t=t, x=x)


def test_rendered_pitch_matches_model(flyover):
    p = flyover
    tt, ff = ridge(p["t"], p["x"], p["fs"], 150.0, 330.0)
    model = doppler_frequency(tt, p["f0"], p["v"], p["d"], p["tau0"], C)
    assert np.median(np.abs(ff - model)) < 0.2  # Hz, 1 s window / 0.5 Hz bins


def test_fit_on_rendered_audio_recovers_geometry(flyover):
    p = flyover
    t1, f1 = ridge(p["t"], p["x"], p["fs"], 150.0, 330.0)
    t2, f2 = ridge(p["t"], p["x"], p["fs"], 340.0, 620.0)
    fit = fit_doppler(
        np.concatenate([t1, t2]),
        np.concatenate([f1, f2]),
        k=np.concatenate([np.ones_like(t1), np.full_like(t2, 2.0)]),
    )
    assert fit.f0 == pytest.approx(p["f0"], rel=0.002)
    assert fit.v == pytest.approx(p["v"], rel=0.05)
    assert fit.d == pytest.approx(p["d"], rel=0.15)
    assert fit.t_cpa == pytest.approx(p["tau0"] + p["d"] / C, abs=1.0)
    assert fit.grade in ("A", "B")


def test_ground_reflection_and_noise_are_optional():
    flight = StraightLineFlight.from_cpa(150.0, 1500.0, 5.0, observer=OBS)
    t, clean = render_audio(flight, OBS, [(100.0, 1.0)], fs=4000.0, duration=10.0, snr_db=None)
    _, refl = render_audio(
        flight, OBS, [(100.0, 1.0)], fs=4000.0, duration=10.0, snr_db=None, ground_reflection=0.9
    )
    assert t.size == clean.size == 40_000
    assert not np.allclose(clean, refl)
