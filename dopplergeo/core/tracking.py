"""Tone tracking: per-frame peaks -> Kalman/Hungarian tracks -> source families.

Key fact used for grouping: every tone from one aircraft gets the *same*
Doppler factor, so the ratio between two of its tones is constant in time even
while both frequencies sweep. Tones from different aircraft (different CPA
times/geometries) have a ratio that changes during the sweep.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .spectral import Spectrogram

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------


@dataclass
class FramePeaks:
    """Peaks of one frame: frequency (Hz), whitened SNR (dB), 1-sigma error (Hz)."""

    f: FloatArray
    snr_db: FloatArray
    sigma_f: FloatArray


SIGMA_FLOOR_CYCLES = 0.03  # floor = 0.03 / T, measured on synthetic chirps (M2)


def frequency_sigma(snr_db: FloatArray, window_s: float) -> FloatArray:
    """1-sigma frequency error of an interpolated spectral peak.

    Cramer-Rao bound for a sinusoid in white noise, expressed with the per-bin
    SNR: sigma ~ sqrt(6) / (2 pi T sqrt(SNR)) ~ 0.39 / (T sqrt(SNR)); x1.5 for
    the Hann window. Added in quadrature to a floor of 0.03 / T that covers
    parabolic-interpolation bias and the tone sweeping inside the window
    (measured: 0.027 Hz at >35 dB with T = 1 s, where the CRB alone gives 0.007).
    """
    snr = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    crb = 0.6 / (window_s * np.sqrt(snr))
    return np.hypot(crb, SIGMA_FLOOR_CYCLES / window_s)


def _interp_peaks(col_db: FloatArray, idx: NDArray[np.intp], freqs: FloatArray) -> FloatArray:
    """Quadratic (parabolic) interpolation of peak positions on a dB column."""
    i = np.clip(idx, 1, col_db.size - 2)
    a, b, c = col_db[i - 1], col_db[i], col_db[i + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return freqs[i] + np.clip(delta, -0.5, 0.5) * (freqs[1] - freqs[0])


def pick_peaks(
    spec: Spectrogram,
    threshold_db: float = 6.0,
    max_per_frame: int = 40,
    white: FloatArray | None = None,
) -> list[FramePeaks | None]:
    """Local maxima of the whitened spectrogram above ``threshold_db``.

    Returns one entry per frame (None for invalid frames). Peaks closer than the
    window resolution are merged (``find_peaks`` distance).
    """
    W = spec.whitened_db() if white is None else white
    db = spec.db
    dist = max(1, int(round(spec.resolution_hz / spec.bin_hz)))
    out: list[FramePeaks | None] = []
    for j in range(W.shape[1]):
        if not spec.frame_valid[j]:
            out.append(None)
            continue
        col = W[:, j]
        idx, props = find_peaks(col, height=threshold_db, distance=dist)
        if idx.size > max_per_frame:
            keep = np.argsort(props["peak_heights"])[-max_per_frame:]
            idx = np.sort(idx[keep])
        snr = col[idx]
        out.append(
            FramePeaks(
                f=_interp_peaks(db[:, j], idx, spec.freqs),
                snr_db=snr,
                sigma_f=frequency_sigma(snr, spec.settings.window_s),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


@dataclass
class ToneTrack:
    id: int
    frame: NDArray[np.intp]
    t: FloatArray
    f: FloatArray
    snr_db: FloatArray
    sigma_f: FloatArray
    manual: bool = False
    family_id: int | None = None
    k: float = 1.0  # multiplier relative to the family's rest frequency

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def median_snr_db(self) -> float:
        return float(np.median(self.snr_db)) if self.snr_db.size else float("-inf")

    @property
    def median_f(self) -> float:
        return float(np.median(self.f))

    def jumps(self, n_mad: float = 8.0) -> int:
        """Count abrupt steps (power change, track switching): points that miss
        the time-weighted interpolation of their two neighbours by more than
        ``n_mad`` robust deviations (gaps in the track are handled)."""
        if self.f.size < 5:
            return 0
        t, f = self.t, self.f
        w = (t[1:-1] - t[:-2]) / (t[2:] - t[:-2])
        e = f[1:-1] - (f[:-2] + w * (f[2:] - f[:-2]))
        mad = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        floor = 3.0 * float(np.median(self.sigma_f))
        return int(np.count_nonzero(np.abs(e) > n_mad * max(mad, floor)))


@dataclass(frozen=True)
class TrackerSettings:
    threshold_db: float = 6.0  # peak height above the whitened floor
    max_peaks_per_frame: int = 40
    sigma_accel: float = 5.0  # Hz/s^2, process noise (frequency acceleration)
    init_rate_sigma: float = 10.0  # Hz/s, prior on a new track's slope
    gate_chi2: float = 9.0  # 3 sigma
    young_penalty: float = 2.0  # extra cost so established tracks win ties
    max_gap_s: float = 1.0
    min_duration_s: float = 5.0
    min_snr_db: float = 8.0  # also the minimum SNR for a peak to *start* a track
    min_fill: float = 0.5  # fraction of frames in a track's span that have a peak
    spawn_exclusion_hz: float = 2.0  # no new track this close to an established one
    max_coast_s: float = 3.0  # longest a pair of crossing tracks may coast without updates
    confirm_s: float = 2.0  # history a track needs before it can coast through a crossing
    # Real tones wander (engine RPM, turbulence): the gate adds this relative
    # jitter to the per-peak Cramer-Rao error, which alone fragments real tracks.
    jitter_rel: float = 0.0
    # Stitching (see ``stitch_tracks``): fragments shorter than ``stitch_min_s``
    # are ignored; a join needs the extrapolated ends to meet within
    # max(stitch_tol_hz, stitch_tol_rel * f) across a gap of at most stitch_max_gap_s.
    stitch: bool = False
    stitch_max_gap_s: float = 3.0
    stitch_min_s: float = 0.5
    stitch_tol_hz: float = 0.5
    stitch_tol_rel: float = 0.003
    stitch_end_s: float = 1.5  # length of the end segments whose slope is extrapolated
    # Linker: "ridge" (dynamic-programming ridge extraction, robust on real
    # recordings) or "kalman" (peak-to-peak Kalman/Hungarian, see link_tracks).
    method: str = "ridge"
    ridge_offset_db: float = 9.0  # whitened level that counts as neutral (noise maxima reach ~+5 dB)
    ridge_cap_db: float = 25.0  # one very strong cell cannot carry a path
    ridge_max_rate_rel: float = 0.25  # |df/dt| limit per Hz of tone (1/s): v^2/(c d) at the CPA
    ridge_max_rate_hz_s: float = 120.0  # ... capped here (faster CPAs split, and are rejoined by fit)
    ridge_min_rate_hz_s: float = 10.0  # ... and at least this much for low tones
    # A passing aircraft's tone only falls; it rises only with RPM changes, so
    # upward steps get a much tighter limit (stops a ridge climbing to another tone).
    ridge_max_rise_rel: float = 0.03
    ridge_min_rise_hz_s: float = 4.0
    ridge_step_penalty: float = 1.0  # a full-rate step costs this many offsets
    ridge_min_score_db_s: float = 15.0  # integral of (level - offset) over the ridge, dB*s
    ridge_suppress_hz: float = 2.0  # around an extracted ridge: max(this, rel * f) + 2 bins
    ridge_suppress_rel: float = 0.006
    ridge_gap_db: float = 3.0  # path points weaker than this are gaps, not data
    max_ridges: int = 24
    # Steady lines (machinery, hum) are removed before ridge extraction by
    # subtracting each bin's running time median over this window; 0 disables.
    stationary_window_s: float = 150.0
    # A passing aircraft's tone always changes pitch: tracks whose 5-95% frequency
    # spread is below this fraction of f are steady lines (hum fragments) and dropped.
    min_sweep_rel: float = 0.01
    # Per-point frequency error of a real tone: the Cramer-Rao error plus this
    # fraction of f (slow source wander, ~0.1-0.3% measured on real flyovers).
    sigma_rel: float = 0.002


def link_tracks(
    peaks: list[FramePeaks | None], times: FloatArray, settings: TrackerSettings = TrackerSettings()
) -> list[ToneTrack]:
    """Associate peaks across frames with a constant-slope Kalman filter per
    track and globally optimal (Hungarian) assignment, gated at ``gate_chi2``.

    State per track: [f, df/dt]; white-acceleration process noise ``sigma_accel``.
    Peaks below ``min_snr_db`` may extend a track but never start one, and no
    track starts within ``spawn_exclusion_hz`` of an established track's
    prediction (sidelobes, two tones merging at a crossing). When two
    confirmed tracks (>= ``confirm_s`` of history) predict within
    ``spawn_exclusion_hz`` of each other they
    *coast* (prediction only, no update) for up to ``max_coast_s``, so the merged
    peak and its interference sidelobes cannot corrupt either slope estimate.
    Gaps are measured
    in *valid* time: invalid (muted) frames, given as ``None``, do not count
    towards ``max_gap_s``. Kalman state is kept in arrays; history in lists.
    """
    s = settings
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    clock = np.cumsum([p is not None for p in peaks]) * hop  # valid-time clock
    c_last = np.zeros(0)
    q = s.sigma_accel**2
    X = np.zeros((0, 2))
    P = np.zeros((0, 2, 2))
    t_last = np.zeros(0)
    hist: list[list[list[float]]] = []  # per active track: [frames, t, f, snr, sigma]
    finished: list[list[list[float]]] = []

    for j, fp in enumerate(peaks):
        t = float(times[j])
        if hist:
            alive = clock[j] - c_last <= s.max_gap_s
            if not np.all(alive):
                finished.extend(h for h, a in zip(hist, alive, strict=True) if not a)
                hist = [h for h, a in zip(hist, alive, strict=True) if a]
                X, P, t_last, c_last = X[alive], P[alive], t_last[alive], c_last[alive]
        if fp is None or fp.f.size == 0:
            continue

        n_act = len(hist)
        assigned = np.zeros(fp.f.size, dtype=bool)
        if n_act:
            dt = t - t_last
            Xp = np.column_stack([X[:, 0] + dt * X[:, 1], X[:, 1]])
            # P' = F P F^T + Q, written out for the 2x2 case.
            p00, p01, p11 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 1]
            Pp = np.empty_like(P)
            Pp[:, 0, 0] = p00 + 2 * dt * p01 + dt * dt * p11 + q * dt**4 / 4
            Pp[:, 0, 1] = Pp[:, 1, 0] = p01 + dt * p11 + q * dt**3 / 2
            Pp[:, 1, 1] = p11 + q * dt * dt
            R = fp.sigma_f**2 + (s.jitter_rel * fp.f) ** 2  # measurement noise
            S = Pp[:, 0, 0][:, None] + R[None, :]
            cost = (fp.f[None, :] - Xp[:, 0][:, None]) ** 2 / S
            young = np.array([len(h[0]) < 5 for h in hist])
            confirmed = np.array([h[1][-1] - h[1][0] >= s.confirm_s for h in hist])
            est_idx = np.nonzero(confirmed)[0]
            coast = np.zeros(n_act, dtype=bool)
            if est_idx.size > 1:
                fe = Xp[est_idx, 0]
                close = np.abs(fe[:, None] - fe[None, :]) < s.spawn_exclusion_hz
                np.fill_diagonal(close, False)
                coast[est_idx[close.any(axis=1)]] = True
                coast &= (t - t_last) <= s.max_coast_s
            cost[young] += s.young_penalty
            limit = s.gate_chi2 + np.where(young, s.young_penalty, 0.0)[:, None]
            cost = np.where(cost <= limit, cost, 1e6)
            cost[coast] = 1e6
            rows, cols = linear_sum_assignment(cost)
            ok = cost[rows, cols] < 1e6
            rows, cols = rows[ok], cols[ok]
            # Kalman update for matched tracks only. Unmatched tracks keep their
            # old state and t_last, so the next prediction spans the whole gap.
            Sm = Pp[rows, 0, 0] + R[cols]
            K = Pp[rows, :, 0] / Sm[:, None]
            innov = fp.f[cols] - Xp[rows, 0]
            X = X.copy()
            P = P.copy()
            X[rows] = Xp[rows] + K * innov[:, None]
            P[rows] = Pp[rows] - K[:, :, None] * Pp[rows, 0, :][:, None, :]
            t_last = t_last.copy()
            t_last[rows] = t
            c_last = c_last.copy()
            c_last[rows] = clock[j]
            c_last[coast] = clock[j]  # coasting tracks stay alive
            for r, c in zip(rows, cols, strict=True):
                h = hist[r]
                h[0].append(j)
                h[1].append(t)
                h[2].append(fp.f[c])
                h[3].append(fp.snr_db[c])
                h[4].append(fp.sigma_f[c])
            assigned[cols] = True

        spawn = ~assigned & (fp.snr_db >= s.min_snr_db)
        if n_act and np.any(spawn):
            est = np.array([len(h[0]) >= 5 for h in hist])
            if np.any(est):
                near = np.abs(fp.f[:, None] - Xp[est, 0][None, :]) < s.spawn_exclusion_hz
                spawn &= ~near.any(axis=1)
        new = np.nonzero(spawn)[0]
        if new.size:
            X = np.vstack([X, np.column_stack([fp.f[new], np.zeros(new.size)])])
            Pn = np.zeros((new.size, 2, 2))
            Pn[:, 0, 0] = fp.sigma_f[new] ** 2 + (s.jitter_rel * fp.f[new]) ** 2
            Pn[:, 1, 1] = s.init_rate_sigma**2
            P = np.concatenate([P, Pn])
            t_last = np.concatenate([t_last, np.full(new.size, t)])
            c_last = np.concatenate([c_last, np.full(new.size, clock[j])])
            for c in new:
                hist.append([[j], [t], [fp.f[c]], [fp.snr_db[c]], [fp.sigma_f[c]]])
    finished.extend(hist)

    raw = [
        ToneTrack(id=-1, frame=np.asarray(h[0], dtype=np.intp), t=np.asarray(h[1]), f=np.asarray(h[2]),
                  snr_db=np.asarray(h[3]), sigma_f=np.asarray(h[4]))
        for h in finished
    ]
    if s.stitch:
        raw = stitch_tracks(raw, s)
    tracks = []
    for tr in raw:
        fill = tr.frame.size / (tr.frame[-1] - tr.frame[0] + 1)
        if (tr.duration >= s.min_duration_s and tr.median_snr_db >= s.min_snr_db and fill >= s.min_fill
                and sweeps(tr, s.min_sweep_rel)):
            tracks.append(tr)
    tracks.sort(key=lambda tr: (tr.t[0], tr.median_f))
    for i, tr in enumerate(tracks):
        tr.id = i
    return tracks


def _end_line(tr: ToneTrack, at_start: bool, span_s: float) -> tuple[float, float, float]:
    """(t, f, slope) of a straight line fitted to the first/last ``span_s`` of a track."""
    if at_start:
        m = tr.t <= tr.t[0] + span_s
        t_ref = float(tr.t[0])
    else:
        m = tr.t >= tr.t[-1] - span_s
        t_ref = float(tr.t[-1])
    tt, ff = tr.t[m], tr.f[m]
    if tt.size < 3 or np.ptp(tt) <= 0:
        return t_ref, float(ff.mean()), 0.0
    slope, icpt = np.polyfit(tt - t_ref, ff, 1)
    return t_ref, float(icpt), float(slope)


def stitch_tracks(tracks: list[ToneTrack], s: TrackerSettings) -> list[ToneTrack]:
    """Join fragments of one tone.

    On real recordings a tone wanders more than the per-peak error allows, so the
    Kalman linker breaks it into pieces that never overlap (and so can never be
    grouped by their constant ratio). Two pieces A -> B are joined when B starts
    at most ``stitch_max_gap_s`` after A ends (or overlaps it), the straight-line
    extrapolations of A's end and B's start meet within the tolerance at the
    midpoint of the gap, and the two slopes are compatible. Pieces that overlap
    in time must agree on the overlap. Greedy in order of mismatch; overlapping
    frames keep the stronger peak.
    """
    pieces = [tr for tr in tracks if tr.duration >= s.stitch_min_s and tr.t.size >= 3]
    rest = [tr for tr in tracks if not (tr.duration >= s.stitch_min_s and tr.t.size >= 3)]
    n = len(pieces)
    if n < 2:
        return tracks
    starts = [_end_line(tr, True, s.stitch_end_s) for tr in pieces]
    ends = [_end_line(tr, False, s.stitch_end_s) for tr in pieces]
    t0 = np.array([tr.t[0] for tr in pieces])
    t1 = np.array([tr.t[-1] for tr in pieces])
    cands = []
    for a in range(n):
        ta, fa, sa = ends[a]
        for b in np.nonzero((t0 > t0[a]) & (t1 > t1[a]) & (t0 - t1[a] <= s.stitch_max_gap_s))[0]:
            tb, fb, sb = starts[b]
            tol = max(s.stitch_tol_hz, s.stitch_tol_rel * 0.5 * (fa + fb))
            if t0[b] <= t1[a]:  # overlap: must coincide where both exist
                A, B = pieces[a], pieces[b]
                lo, hi = t0[b], t1[a]
                ia = (A.t >= lo) & (A.t <= hi)
                ib = (B.t >= lo) & (B.t <= hi)
                if ia.sum() < 2 or ib.sum() < 2:
                    continue
                mis = float(np.median(np.abs(np.interp(B.t[ib], A.t[ia], A.f[ia]) - B.f[ib])))
            else:
                tm = 0.5 * (ta + tb)
                mis = abs((fa + sa * (tm - ta)) - (fb + sb * (tm - tb)))
                slope_tol = max(2.0, 0.5 * max(abs(sa), abs(sb)))
                if abs(sa - sb) > slope_tol:
                    continue
            if mis <= tol:
                cands.append((mis / tol, a, int(b)))
    cands.sort()
    nxt = [-1] * n
    prv = [-1] * n
    for _, a, b in cands:
        if nxt[a] == -1 and prv[b] == -1:
            # no cycles possible: b always starts after a
            nxt[a], prv[b] = b, a
    out = list(rest)
    for i in range(n):
        if prv[i] != -1:
            continue
        chain = [pieces[i]]
        j = nxt[i]
        while j != -1:
            chain.append(pieces[j])
            j = nxt[j]
        out.append(merge_tracks(chain) if len(chain) > 1 else chain[0])
    return out


# ---------------------------------------------------------------------------
# Ridge extraction (dynamic programming)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, nogil=True, parallel=True)
def _ridge_dp_frames(W, penT, ks, ilo, ihi, prev, j0, D, ptr, start):  # pragma: no cover (compiled)
    """Frames j0.. of the ridge DP, compiled: per bin b, the best predecessor
    prev[b - ks[i]] - penT[b, i] over the steps i in [ilo[b], ihi[b]) (finite
    penalty, source bin in range). Same arithmetic as the former vectorised
    version (float64, ties to the smallest k as argmax took the first maximum)
    without its (n_steps x n_bins) temporary per frame, which made the tracker
    memory-bandwidth bound (~80% of analysis time)."""
    nb = W.shape[0]
    nf = W.shape[1]
    cur = np.empty(nb)
    for j in range(j0, nf):
        for b in numba.prange(nb):  # bins are independent within a frame: threads
            best = -np.inf
            bi = ilo[b]
            for i in range(ilo[b], ihi[b]):
                v = prev[b - ks[i]] - penT[b, i]
                if v > best:
                    best = v
                    bi = i
            st = not (best > 0)
            c = np.float64(W[b, j]) + (0.0 if st else best)
            cur[b] = c
            D[j, b] = c
            ptr[j, b] = ks[bi]
            start[j, b] = st
        prev[:] = cur


class _RidgeDP:
    """Local (start/stop anywhere) Viterbi over the whitened spectrogram.

    Cell score = min(level, cap) - offset, so noise sums negative and a tone,
    even a weak, wandering one, sums positive along its path. A step of k bins
    between frames costs ``penalty * offset * (k / K(f))**2`` with K(f) the rate
    limit at that frequency. Each extraction takes the best-scoring segment,
    suppresses a band around it and recomputes only the frames from the
    segment's start onwards (earlier frames are unaffected).
    """

    def __init__(self, score: FloatArray, freqs: FloatArray, hop: float, s: TrackerSettings):
        self.W = score
        nb, self.nf = score.shape
        bin_hz = float(freqs[1] - freqs[0])
        rate = np.clip(s.ridge_max_rate_rel * freqs, s.ridge_min_rate_hz_s, s.ridge_max_rate_hz_s)
        rise = np.minimum(np.maximum(s.ridge_max_rise_rel * freqs, s.ridge_min_rise_hz_s), rate)
        self.K = np.maximum(1, np.ceil(rate * hop / bin_hz)).astype(int)  # falling, per destination bin
        K_up = np.maximum(1, np.ceil(rise * hop / bin_hz)).astype(int)  # rising
        self.kmax = int(self.K.max())
        ks = np.arange(-self.kmax, self.kmax + 1)  # predecessor at b - k: k > 0 is a rise
        lim = np.where(ks[:, None] > 0, K_up[None, :], self.K[None, :])
        pen = s.ridge_step_penalty * s.ridge_offset_db * (ks[:, None] / lim) ** 2
        self.pen = np.where(np.abs(ks[:, None]) <= lim, pen, np.inf)
        self.ks = ks
        self.D = np.zeros((self.nf, nb), dtype=np.float32)
        self.ptr = np.zeros((self.nf, nb), dtype=np.int16)
        self.start = np.zeros((self.nf, nb), dtype=bool)
        self.run(0)

    def run(self, j0: int) -> None:
        nb = self.W.shape[0]
        if not hasattr(self, "_penT"):
            # Per destination bin, the contiguous step range with a finite penalty
            # and a source bin b - k inside the spectrum (ks ascends, so the first
            # index in range is the smallest k: argmax's tie-break).
            src = np.arange(nb)[None, :] - self.ks[:, None]
            ok = np.isfinite(self.pen) & (src >= 0) & (src < nb)
            any_ok = ok.any(0)
            self._ilo = np.where(any_ok, ok.argmax(0), 0).astype(np.int64)
            self._ihi = np.where(any_ok, ok.shape[0] - ok[::-1].argmax(0), 0).astype(np.int64)
            assert all(ok[self._ilo[b]:self._ihi[b], b].all() for b in range(nb))  # contiguous
            self._penT = np.ascontiguousarray(self.pen.T)
            self._ks64 = self.ks.astype(np.int64)
        prev = self.D[j0 - 1].astype(np.float64) if j0 > 0 else np.zeros(nb)
        _ridge_dp_frames(self.W, self._penT, self._ks64, self._ilo, self._ihi, prev, j0, self.D, self.ptr,
                         self.start)

    def best(self) -> tuple[float, NDArray[np.intp], NDArray[np.intp]]:
        j, b = np.unravel_index(int(np.argmax(self.D)), self.D.shape)
        score = float(self.D[j, b])
        bins, frames = [int(b)], [int(j)]
        while not self.start[j, b] and j > 0:
            b = b - int(self.ptr[j, b])
            j -= 1
            bins.append(int(b))
            frames.append(int(j))
        return score, np.asarray(bins[::-1]), np.asarray(frames[::-1])


def ridge_tracks(spec: Spectrogram, white: FloatArray | None = None,
                 settings: TrackerSettings = TrackerSettings()) -> list[ToneTrack]:
    """Tone tracks by repeated best-ridge extraction (see ``_RidgeDP``).

    Each path is refined to the local maximum within +-2 bins per frame and
    interpolated (parabolic, on dB). Points weaker than ``ridge_gap_db`` are
    dropped as gaps. Ridges below ``ridge_min_score_db_s`` end the search.
    """
    s = settings
    W = spec.whitened_db() if white is None else white
    db = spec.db
    hop = float(np.median(np.diff(spec.times))) if spec.times.size > 1 else 1.0
    score = (np.minimum(W, s.ridge_cap_db) - s.ridge_offset_db).astype(np.float32)
    score[:, ~spec.frame_valid] = -s.ridge_offset_db
    if score.shape[1] < 2 or score.shape[0] < 5:
        return []
    dp = _RidgeDP(score, spec.freqs, hop, s)
    bin_hz = spec.bin_hz
    nb = W.shape[0]
    raw: list[ToneTrack] = []
    for _ in range(s.max_ridges):
        sc, bins, frames = dp.best()
        if sc * hop < s.ridge_min_score_db_s:
            break
        # Suppress the ridge (and its spectral leakage) for later extractions.
        # A sweeping tone is smeared over |df/dt| * window, so widen by that.
        fpath = spec.freqs[bins]
        rate = np.abs(np.gradient(fpath, spec.times[frames])) if bins.size > 1 else np.zeros(1)
        k_sm = max(1, int(round(spec.settings.window_s / hop)))
        rate = np.convolve(rate, np.ones(k_sm) / k_sm, mode="same")  # path steps are quantised
        for b, j, r in zip(bins, frames, rate, strict=True):
            half = max(s.ridge_suppress_hz, s.ridge_suppress_rel * spec.freqs[b]) + 0.5 * r * spec.settings.window_s
            w = 2 + int(np.ceil(half / bin_hz))
            # ... and further over the whole peak: a strong tone's skirt stays
            # above the noise offset well beyond the main lobe ("shadow" ridges).
            cap = w + int(np.ceil(0.05 * spec.freqs[b] / bin_hz))
            lo_b, hi_b = max(0, b - w), min(nb, b + w + 1)
            col = score[:, j]
            # Descend the skirt only: stop at the valley before a neighbouring tone.
            while lo_b > 0 and b - lo_b < cap and 0 < col[lo_b - 1] <= col[lo_b] + 1.0:
                lo_b -= 1
            while hi_b < nb and hi_b - b < cap and 0 < col[hi_b] <= col[hi_b - 1] + 1.0:
                hi_b += 1
            score[lo_b:hi_b, j] = -s.ridge_offset_db
        dp.run(int(frames[0]))
        # Refine to the local maximum and drop weak points (gaps).
        lo = np.clip(bins - 2, 0, nb - 1)
        idx = np.array([lo[i] + int(np.argmax(W[lo[i] : min(nb, bins[i] + 3), frames[i]]))
                        for i in range(bins.size)])
        snr = W[idx, frames]
        keep = (snr >= s.ridge_gap_db) & spec.frame_valid[frames]
        if keep.sum() < 3:
            continue
        idx, fr, snr = idx[keep], frames[keep], snr[keep]
        f = np.array([_interp_peaks(db[:, j], np.array([i]), spec.freqs)[0] for i, j in zip(idx, fr, strict=True)])
        sig = np.hypot(frequency_sigma(snr, spec.settings.window_s), s.sigma_rel * f)
        raw.append(ToneTrack(id=-1, frame=fr.astype(np.intp), t=spec.times[fr], f=f, snr_db=snr, sigma_f=sig))
    tracks = []
    for tr in raw:
        fill = tr.frame.size / (tr.frame[-1] - tr.frame[0] + 1)
        if (tr.duration >= s.min_duration_s and tr.median_snr_db >= s.min_snr_db and fill >= s.min_fill
                and sweeps(tr, s.min_sweep_rel)):
            tracks.append(tr)
    tracks.sort(key=lambda tr: (tr.t[0], tr.median_f))
    for i, tr in enumerate(tracks):
        tr.id = i
    return tracks


def track_from_guide(
    spec: Spectrogram,
    guide_t: FloatArray,
    guide_f: FloatArray,
    half_width_hz: float = 5.0,
    track_id: int = -1,
    min_snr_db: float = 3.0,
) -> ToneTrack:
    """Manual mode: follow the strongest whitened peak within +-half_width_hz of
    a user-drawn polyline (guide_t, guide_f)."""
    W = spec.whitened_db()
    db = spec.db
    order = np.argsort(guide_t)
    gt, gf = np.asarray(guide_t, float)[order], np.asarray(guide_f, float)[order]
    rows = []
    for j, t in enumerate(spec.times):
        if not spec.frame_valid[j] or t < gt[0] or t > gt[-1]:
            continue
        fc = float(np.interp(t, gt, gf))
        band = np.nonzero(np.abs(spec.freqs - fc) <= half_width_hz)[0]
        if band.size < 3:
            continue
        i = band[np.argmax(W[band, j])]
        if W[i, j] < min_snr_db:
            continue
        f = float(_interp_peaks(db[:, j], np.array([i]), spec.freqs)[0])
        rows.append((j, t, f, W[i, j]))
    if not rows:
        raise ValueError("no peaks found along the guide")
    j, t, f, snr = (np.asarray(v) for v in zip(*rows, strict=True))
    return ToneTrack(
        id=track_id,
        frame=j.astype(np.intp),
        t=t.astype(float),
        f=f.astype(float),
        snr_db=snr.astype(float),
        sigma_f=frequency_sigma(snr, spec.settings.window_s),
        manual=True,
    )


# ---------------------------------------------------------------------------
# Families (tones from one source)
# ---------------------------------------------------------------------------


def sweeps(tr: ToneTrack, min_rel: float) -> bool:
    """True when the track's frequency changes by at least ``min_rel`` (robust
    5-95% spread relative to its median): it can carry Doppler information."""
    if min_rel <= 0 or tr.f.size < 3:
        return True
    lo, hi = np.percentile(tr.f, [5, 95])
    return (hi - lo) >= min_rel * float(np.median(tr.f))


@dataclass
class Family:
    id: int
    tracks: list[ToneTrack] = field(default_factory=list)

    def stacked(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """All points of the family: t, f, k, sigma_f (for ``fit_doppler``)."""
        t = np.concatenate([tr.t for tr in self.tracks])
        f = np.concatenate([tr.f for tr in self.tracks])
        k = np.concatenate([np.full(tr.t.size, tr.k) for tr in self.tracks])
        s = np.concatenate([tr.sigma_f for tr in self.tracks])
        return t, f, k, s

    @property
    def ks(self) -> list[float]:
        return [tr.k for tr in self.tracks]

    @property
    def t_range(self) -> tuple[float, float]:
        return (min(tr.t[0] for tr in self.tracks), max(tr.t[-1] for tr in self.tracks))


def _pair_ratio(a: ToneTrack, b: ToneTrack, min_overlap: int,
                tol: float = 0.005) -> tuple[float, float, float] | None:
    """(median f_b/f_a, fraction of common frames whose ratio is within ``tol``
    of the median, relative sweep of a over those frames), or None when they
    overlap too little. The fraction (not a percentile spread) lets a pair
    survive a few seconds of corruption where one tone passes within the
    resolution of another aircraft's tone."""
    common, ia, ib = np.intersect1d(a.frame, b.frame, return_indices=True)
    if common.size < min_overlap:
        return None
    r = b.f[ib] / a.f[ia]
    med = float(np.median(r))
    ok = np.abs(r / med - 1.0) <= tol
    fa = a.f[ia][ok]
    if fa.size < min_overlap:
        return med, float(ok.mean()), 0.0
    sweep = float((fa.max() - fa.min()) / np.median(fa))
    return med, float(ok.mean()), sweep


def group_families(
    tracks: list[ToneTrack],
    hop_s: float,
    min_overlap_s: float = 3.0,
    ratio_spread_tol: float = 0.01,
    min_sweep: float = 0.01,
    min_ratio_inliers: float = 0.6,
    integer_tol: float = 0.01,
    max_denominator: int = 4,
) -> list[Family]:
    """Group tracks whose frequency ratio stays constant while they sweep.

    Two tracks join when, over >= ``min_overlap_s`` of common frames, at least
    ``min_ratio_inliers`` of the frames have f_b/f_a within +-``ratio_spread_tol``/2
    of its median *and* the tone actually sweeps by >= ``min_sweep`` over them
    (flat tones carry no evidence). Each family's
    lowest track is the reference; members get k = ratio, snapped to integers
    when they are harmonics (allowing a missing fundamental: 2:3 -> k = 2, 3).
    """
    n = len(tracks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    min_overlap = max(3, int(round(min_overlap_s / hop_s)))
    ratio: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pr = _pair_ratio(tracks[i], tracks[j], min_overlap, tol=ratio_spread_tol / 2)
            if pr is None:
                continue
            med, frac, sweep = pr
            if frac >= min_ratio_inliers and sweep >= min_sweep:
                ratio[(i, j)] = med
                ratio[(j, i)] = 1.0 / med
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    families = []
    for fid, members in enumerate(sorted(groups.values(), key=lambda m: min(tracks[i].t[0] for i in m))):
        base = min(members, key=lambda i: tracks[i].median_f)
        # Ratios to the base via BFS over pair links.
        rel = {base: 1.0}
        queue = deque([base])
        while queue:
            u = queue.popleft()
            for v in members:
                if v not in rel and (u, v) in ratio:
                    rel[v] = rel[u] * ratio[(u, v)]
                    queue.append(v)
        ks = np.array([rel[i] for i in members])
        for m in range(1, max_denominator + 1):
            scaled = ks * m
            if np.all(np.abs(scaled - np.round(scaled)) / scaled < integer_tol):
                ks = np.round(scaled)
                break
        fam = Family(id=fid)
        by_k: dict[float, list[ToneTrack]] = {}
        for i, k in zip(members, ks, strict=True):
            tracks[i].family_id = fid
            tracks[i].k = float(k)
            by_k.setdefault(round(float(k), 3), []).append(tracks[i])
        fam.tracks = _drop_ratio_outliers(
            sorted((merge_tracks(g) for g in by_k.values()), key=lambda tr: tr.k), ratio_spread_tol / 2
        )
        families.append(fam)
    return families


def _drop_ratio_outliers(tracks: list[ToneTrack], tol: float) -> list[ToneTrack]:
    """Within one family every tone gives the same f/k at the same frame. Drop
    the points that disagree with the per-frame median by more than ``tol``
    (a tone briefly captured by another aircraft's tone at a near-crossing);
    with only two tones present both points of a disagreeing frame go."""
    if len(tracks) < 2:
        return tracks
    frames = np.unique(np.concatenate([tr.frame for tr in tracks]))
    vals = np.full((len(tracks), frames.size), np.nan)
    for i, tr in enumerate(tracks):
        vals[i, np.searchsorted(frames, tr.frame)] = tr.f / tr.k
    n = np.sum(np.isfinite(vals), axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, axis=0)
    out = []
    for i, tr in enumerate(tracks):
        j = np.searchsorted(frames, tr.frame)
        bad = (n[j] >= 2) & (np.abs(tr.f / tr.k / med[j] - 1.0) > tol)
        keep = ~bad
        if keep.sum() < 3:
            out.append(tr)
            continue
        out.append(ToneTrack(id=tr.id, frame=tr.frame[keep], t=tr.t[keep], f=tr.f[keep], snr_db=tr.snr_db[keep],
                             sigma_f=tr.sigma_f[keep], manual=tr.manual, family_id=tr.family_id, k=tr.k))
    return out


def merge_tracks(group: list[ToneTrack]) -> ToneTrack:
    """Join pieces of one tone (same family, same k) into a single track. Where
    two pieces share a frame, the higher-SNR peak is kept."""
    if len(group) == 1:
        return group[0]
    frame = np.concatenate([tr.frame for tr in group])
    snr = np.concatenate([tr.snr_db for tr in group])
    order = np.lexsort((-snr, frame))  # by frame, strongest first
    frame = frame[order]
    keep = np.ones(frame.size, dtype=bool)
    keep[1:] = frame[1:] != frame[:-1]
    sel = order[keep]
    cat = lambda name: np.concatenate([getattr(tr, name) for tr in group])[sel]  # noqa: E731
    first = min(group, key=lambda tr: tr.id)
    return ToneTrack(
        id=first.id,
        frame=frame[keep],
        t=cat("t"),
        f=cat("f"),
        snr_db=cat("snr_db"),
        sigma_f=cat("sigma_f"),
        manual=any(tr.manual for tr in group),
        family_id=first.family_id,
        k=first.k,
    )
