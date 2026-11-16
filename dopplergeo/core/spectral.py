"""Pre-processing and time-frequency representation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT, butter, sosfiltfilt
from scipy.signal.windows import hann

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpectroSettings:
    window_s: float = 1.0  # ~1 Hz resolution for slow low tones
    hop_s: float = 0.1  # 10 frames/s
    zero_pad: int = 2  # FFT length = zero_pad * window
    fmin: float = 20.0
    fmax: float = 5000.0
    whiten_bins: int = 61  # running-median width across frequency


@dataclass
class Spectrogram:
    times: FloatArray  # frame centres, media time (s)
    freqs: FloatArray  # bin centres (Hz)
    power: FloatArray  # (n_freqs, n_frames), linear power
    settings: SpectroSettings
    frame_valid: NDArray[np.bool_]  # False for muted / edge frames

    @property
    def db(self) -> FloatArray:
        return 10.0 * np.log10(self.power + 1e-20)

    @property
    def bin_hz(self) -> float:
        return float(self.freqs[1] - self.freqs[0])

    @property
    def resolution_hz(self) -> float:
        """Window resolution 1/T (wider than the zero-padded bin spacing)."""
        return 1.0 / self.settings.window_s

    def whitened_db(self) -> FloatArray:
        """dB above a running median across frequency, per frame. Tones stand out
        as positive peaks; broadband noise (wind, traffic) is flattened to ~0 dB."""
        db = self.db
        return db - running_median_floor(db, self.settings.whiten_bins)


def running_median_floor(db: FloatArray, width: int, step: int | None = None, chunk: int = 256) -> FloatArray:
    """Running median across frequency (axis 0), evaluated every ``step`` bins
    (default width // 4) and linearly interpolated between. ~10x faster than a
    full ``median_filter`` and the floor is smooth on that scale anyway."""
    n, m = db.shape
    if n <= width:
        return np.broadcast_to(np.median(db, axis=0), db.shape).copy()
    step = step or max(1, width // 4)
    pad = width // 2
    dbp = np.pad(db.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    centers = np.arange(0, n, step)
    if centers[-1] != n - 1:
        centers = np.append(centers, n - 1)
    floor_c = np.empty((centers.size, m), dtype=np.float32)
    for a in range(0, m, chunk):
        win = sliding_window_view(dbp[:, a : a + chunk], width, axis=0)[centers]
        floor_c[:, a : a + chunk] = np.median(win, axis=-1)
    idx = np.arange(n)
    seg = np.clip(np.searchsorted(centers, idx, side="right") - 1, 0, centers.size - 2)
    w = ((idx - centers[seg]) / (centers[seg + 1] - centers[seg]))[:, None]
    return (1 - w) * floor_c[seg] + w * floor_c[seg + 1]


def highpass(x: NDArray, sample_rate: float, cutoff_hz: float = 15.0, order: int = 4) -> FloatArray:
    """Zero-phase Butterworth high-pass (removes DC and handling rumble)."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def mute_mask(times: FloatArray, mutes: Sequence[tuple[float, float]], pad: float = 0.0) -> NDArray[np.bool_]:
    """True where ``times`` fall outside every muted [start, end) range (+-pad)."""
    ok = np.ones(times.shape, dtype=bool)
    for a, b in mutes:
        ok &= ~((times >= a - pad) & (times < b + pad))
    return ok


def compute_spectrogram(
    x: NDArray,
    sample_rate: float,
    settings: SpectroSettings = SpectroSettings(),
    t0: float = 0.0,
    mutes: Sequence[tuple[float, float]] = (),
) -> Spectrogram:
    """Hann-window STFT, power, cropped to [fmin, fmax].

    ``t0`` is the media time of ``x[0]``. Frames whose window overlaps a muted
    range or runs past the signal edges are flagged invalid (kept in the array so
    the display stays continuous).
    """
    x = np.asarray(x, dtype=float)
    nwin = int(round(settings.window_s * sample_rate))
    hop = max(1, int(round(settings.hop_s * sample_rate)))
    stft = ShortTimeFFT(
        hann(nwin, sym=False), hop=hop, fs=sample_rate, mfft=settings.zero_pad * nwin, scale_to="psd"
    )
    S = stft.stft(x)
    power = np.abs(S) ** 2
    freqs = stft.f
    times = stft.t(x.size) + t0
    band = (freqs >= settings.fmin) & (freqs <= settings.fmax)
    half = settings.window_s / 2
    valid = (times - half >= t0) & (times + half <= t0 + x.size / sample_rate)
    valid &= mute_mask(times, mutes, pad=half)
    return Spectrogram(times, freqs[band], power[band], settings, valid)


def subtract_noise(spec: Spectrogram, noise_range: tuple[float, float], over: float = 1.0) -> Spectrogram:
    """Spectral subtraction: remove the mean noise power per bin measured over a
    user-selected noise-only time range (floored at 1% of the original)."""
    sel = (spec.times >= noise_range[0]) & (spec.times <= noise_range[1]) & spec.frame_valid
    if not np.any(sel):
        raise ValueError("noise range contains no valid frames")
    noise = spec.power[:, sel].mean(axis=1, keepdims=True)
    power = np.maximum(spec.power - over * noise, 0.01 * spec.power)
    return Spectrogram(spec.times, spec.freqs, power, spec.settings, spec.frame_valid)


def remove_stationary_lines(white: FloatArray, hop_s: float, window_s: float = 150.0, block_s: float = 10.0,
                            valid: NDArray[np.bool_] | None = None) -> FloatArray:
    """Subtract each frequency bin's running median over time (``window_s``).

    Steady lines from local machinery, mains hum or wind-turbine drives sit in
    the same bin most of the time, so their time median is high; an aircraft
    tone sweeps through a bin in seconds, even on a slow approach, so its bin's
    median stays at the noise floor. Only positive baselines are removed.
    Implemented as medians of ``block_s`` blocks, then a median across the
    blocks within the window, interpolated per frame (fast on long clips).
    """
    n, m = white.shape
    if m < 3 or window_s <= 0:
        return white
    x = white.astype(np.float32, copy=True)
    if valid is not None and not valid.all():
        x[:, ~valid] = np.nan
    bf = max(1, int(round(block_s / hop_s)))
    nb = int(np.ceil(m / bf))
    pad = nb * bf - m
    xp = np.pad(x, ((0, 0), (0, pad)), constant_values=np.nan) if pad else x
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (muted) blocks
            blocks = np.nanmedian(xp.reshape(n, nb, bf), axis=2)          # (n, nb)
            k = max(1, int(round(window_s / block_s / 2)))
            if nb <= 2 * k + 1:
                base_b = np.repeat(np.nanmedian(blocks, axis=1, keepdims=True), nb, axis=1)
            else:
                padded = np.pad(blocks, ((0, 0), (k, k)), mode="edge")
                base_b = np.nanmedian(sliding_window_view(padded, 2 * k + 1, axis=1), axis=-1)
    base_b = np.nan_to_num(np.maximum(base_b, 0.0), nan=0.0)
    centers = (np.arange(nb) + 0.5) * bf - 0.5
    frames = np.arange(m)
    base = np.empty((n, m), dtype=np.float32)
    for i in range(0, n, 2048):  # interpolate per bin along time, in chunks of bins
        sl = slice(i, min(n, i + 2048))
        idx = np.clip(np.searchsorted(centers, frames, side="right") - 1, 0, max(nb - 2, 0))
        if nb == 1:
            base[sl] = base_b[sl, :1]
            continue
        w = np.clip((frames - centers[idx]) / (centers[idx + 1] - centers[idx]), 0.0, 1.0)
        base[sl] = (1 - w) * base_b[sl, idx] + w * base_b[sl, idx + 1]
    return white - base
