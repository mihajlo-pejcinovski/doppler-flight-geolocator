"""Audio -> spectrogram -> tone tracks -> families -> Doppler fits."""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .atmosphere import SPEED_OF_SOUND_15C
from .doppler import DopplerFit, fit_doppler
from .media import DEFAULT_SAMPLE_RATE, AudioClip, load_audio
from .spectral import (
    SpectroSettings,
    Spectrogram,
    compute_spectrogram,
    highpass,
    remove_stationary_lines,
    subtract_noise,
)
from .tracking import (
    Family,
    ToneTrack,
    TrackerSettings,
    group_families,
    link_tracks,
    merge_tracks,
    pick_peaks,
    ridge_tracks,
)


@dataclass
class FamilyResult:
    family: Family
    fit: DopplerFit | None
    error: str | None = None

    @property
    def flags(self) -> dict[str, bool | int]:
        """GUI quality flags (spec section 'Quality flags')."""
        jumps = sum(tr.jumps() for tr in self.family.tracks)
        if self.fit is None:
            return {"cpa_covered": False, "asymptotes_visible": False, "jumps": jumps}
        f = self.fit
        # Slope magnitude at the ends vs the CPA slope: < 20% means the curve has
        # flattened towards its asymptote.
        t0, t1 = self.family.t_range
        h = 0.05
        slope = lambda t: (f.predict([t + h])[0] - f.predict([t - h])[0]) / (2 * h)  # noqa: E731
        s_cpa = abs(slope(f.t_cpa))
        flat = s_cpa > 0 and abs(slope(t0)) < 0.2 * s_cpa and abs(slope(t1)) < 0.2 * s_cpa
        return {"cpa_covered": f.cpa_covered, "asymptotes_visible": bool(flat), "jumps": jumps}


@dataclass
class AnalysisResult:
    spectrogram: Spectrogram
    tracks: list[ToneTrack]
    families: list[FamilyResult]
    t0: float  # media time of the first sample
    clip: AudioClip | None = field(default=None, repr=False)

    @property
    def reliable(self) -> list[FamilyResult]:
        """Families graded A or B. The rest are shown as 'unexplained'."""
        return [fr for fr in self.families if fr.fit is not None and fr.fit.grade in ("A", "B")]


def _fit_one(fam: Family, c: float, trim: bool = True, fixed: dict[str, float] | None = None) -> FamilyResult:
    t, f, k, s = fam.stacked()
    try:
        fit = fit_doppler(t, f, k=k, sigma=s, c=c, fixed=fixed)
    except (ValueError, np.linalg.LinAlgError) as e:
        return FamilyResult(fam, None, str(e))
    if trim and not fixed and fit.grade != "A":
        fam, fit = _trim_steps(fam, fit, c)
    return FamilyResult(fam, fit)


STEP_REL = 0.025        # a step: frequency changes by > 2.5 % ...
STEP_SPAN_S = 1.5       # ... within this time ...
FLAT_SPAN_S = 3.0       # ... with the track flat (slope < STEP_FLAT x the step's) this long on both sides
STEP_FLAT = 0.3
MAX_TRIMS = 3
MAX_STEP_CANDIDATES = 6


def _track_steps(tr: ToneTrack) -> list[tuple[float, int]]:
    """Sudden steps in one track, strongest first: (relative size, index of the first
    point after the step). On real recordings a ridge in the broadband CPA roar, or
    at its end, jumps onto another tone (a harmonic, the next aircraft's approach)
    and stays there; a single Doppler curve cannot do that, but a close pass's own
    CPA drop looks the same locally, so steps are only cut where it improves the fit."""
    t, f = tr.t, tr.f
    if t.size < 10:
        return []
    out: list[tuple[float, int]] = []
    ib = np.searchsorted(t, t + STEP_SPAN_S)
    ok = ib < t.size
    rel = np.zeros(t.size)
    rel[ok] = (f[ib[ok]] - f[ok]) / f[ok]
    for a in np.flatnonzero(np.abs(rel) >= STEP_REL):
        b = int(ib[a])
        df = rel[a]
        step_rate = abs(f[b] - f[a]) / max(t[b] - t[a], 1e-3)
        pre = (t >= t[a] - FLAT_SPAN_S) & (t <= t[a])
        post = (t >= t[b]) & (t <= t[b] + FLAT_SPAN_S)
        if pre.sum() < 5 or post.sum() < 5:
            continue
        slope = lambda m: abs(np.polyfit(t[m], f[m], 1)[0])  # noqa: E731
        if slope(pre) < STEP_FLAT * step_rate and slope(post) < STEP_FLAT * step_rate:
            mid = a + int(np.argmax(np.abs(np.diff(f[a:b + 1])))) + 1  # the steepest point inside the step
            if not out or mid - out[-1][1] > 3:
                out.append((abs(df), mid))
            elif abs(df) > out[-1][0]:
                out[-1] = (abs(df), mid)
    return sorted(out, reverse=True)


def _subtrack(tr: ToneTrack, sl: slice) -> ToneTrack:
    return ToneTrack(tr.id, tr.frame[sl], tr.t[sl], tr.f[sl], tr.snr_db[sl], tr.sigma_f[sl], tr.manual,
                     tr.family_id, tr.k)


def _fit_key(fit: DopplerFit) -> tuple[int, float]:
    return "ABC".index(fit.grade), fit.chi2_red


def _trim_steps(fam: Family, fit: DopplerFit, c: float) -> tuple[Family, DopplerFit]:
    """Cut tracks at sudden steps when that makes the family fit one Doppler curve
    clearly better: the trimmed fit must still contain its CPA, and grade better or
    halve chi2_red at the same grade. Greedy, at most MAX_TRIMS cuts; manual
    tracks are never cut."""
    for _ in range(MAX_TRIMS):
        cands = [(size, ti, idx) for ti, tr in enumerate(fam.tracks) if not tr.manual
                 for size, idx in _track_steps(tr)]
        cands = sorted(cands, reverse=True)[:MAX_STEP_CANDIDATES]
        best = None
        for _, ti, idx in cands:
            tr = fam.tracks[ti]
            for sl in (slice(0, idx), slice(idx, None)):  # keep the part before, or after, the step
                part = _subtrack(tr, sl)
                if part.t.size < 10:
                    continue
                trial = Family(fam.id, fam.tracks[:ti] + [part] + fam.tracks[ti + 1:])
                tt, ff, kk, ss = trial.stacked()
                try:
                    tf = fit_doppler(tt, ff, k=kk, sigma=ss, c=c)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not (tf.success and tf.cpa_covered):
                    continue
                g_new, g_old = _fit_key(tf)[0], _fit_key(fit)[0]
                better = g_new < g_old or (g_new == g_old and tf.chi2_red < 0.5 * fit.chi2_red)
                if better and (best is None or _fit_key(tf) < _fit_key(best[1])):
                    best = (trial, tf)
        if best is None:
            break
        fam, fit = best
        if fit.grade == "A":
            break
    return fam, fit


def _rest_ratio(fa: float, fb: float, tol: float, max_den: int = 4) -> float | None:
    """fb/fa if it is a small rational p/q (q <= max_den) within ``tol``."""
    r = fb / fa
    for q in range(1, max_den + 1):
        p = round(r * q)
        if p >= 1 and abs(r - p / q) / r < tol:
            return p / q
    return None


def _consistent(a: DopplerFit, b: DopplerFit) -> float | None:
    """If two family fits describe the same aircraft, return f0_b / f0_a."""
    def close(name: str, rel: float, abs_: float = 0.0) -> bool:
        va, vb = getattr(a, name), getattr(b, name)
        tol = 3.0 * float(np.hypot(a.stderr(name), b.stderr(name))) + rel * abs(va) + abs_
        return abs(va - vb) <= tol

    if not (close("v", 0.02) and close("d", 0.05)):
        return None
    if abs(a.t_cpa - b.t_cpa) > 3.0 * float(np.hypot(a.stderr("t_cpa"), b.stderr("t_cpa"))) + 1.0:
        return None
    return _rest_ratio(a.f0, b.f0, tol=0.005)


def merge_consistent_families(results: list[FamilyResult], c: float) -> list[FamilyResult]:
    """Join families whose single-pass fits agree (same v, d, t_cpa and rest
    frequencies in a small-integer ratio). This reunites tones that broke where
    they crossed another aircraft's tones, where the ratio test has no overlap."""
    results = list(results)
    merged = True
    while merged:
        merged = False
        good = [i for i, r in enumerate(results) if r.fit is not None and r.fit.grade in ("A", "B")]
        for ii, i in enumerate(good):
            for j in good[ii + 1 :]:
                ratio = _consistent(results[i].fit, results[j].fit)
                if ratio is None:
                    continue
                fa, fb = results[i].family, results[j].family
                # Express b's k values relative to a's rest frequency.
                for tr in fb.tracks:
                    tr.k = tr.k * ratio
                _join(fa, fb.tracks)
                results[i] = _fit_one(fa, c)
                del results[j]
                merged = True
                break
            if merged:
                break
    return results


def _join(anchor: Family, tracks: list[ToneTrack]) -> None:
    by_k: dict[float, list[ToneTrack]] = {}
    for tr in anchor.tracks + tracks:
        tr.family_id = anchor.id
        by_k.setdefault(round(tr.k, 3), []).append(tr)
    anchor.tracks = sorted((merge_tracks(g) for g in by_k.values()), key=lambda tr: tr.k)


def absorb_by_model(
    results: list[FamilyResult],
    c: float,
    spread_tol: float = 0.005,
    min_sweep: float = 0.01,
    integer_tol: float = 0.01,
) -> list[FamilyResult]:
    """Attach low-grade families to a well-fitted one when every track keeps a
    constant ratio to the fitted curve while it sweeps.

    This is the constant-ratio rule of ``group_families`` with the fitted model
    standing in for an overlapping track, so it reunites pieces that never
    overlap (an approach segment cut off by a mute, or by smearing at a fast
    CPA). k is the ratio, snapped to an integer when within ``integer_tol``.
    """
    results = list(results)
    anchors = sorted(
        (r for r in results if r.fit is not None and r.fit.grade in ("A", "B")),
        key=lambda r: (r.fit.grade, -sum(tr.t.size for tr in r.family.tracks)),
    )
    absorbed: set[int] = set()
    for idx, r in enumerate(results):
        if r in anchors:
            continue
        for a in anchors:
            ks = []
            for tr in r.family.tracks:
                pred = a.fit.predict(tr.t, 1.0)
                ratio = tr.f / pred
                med = float(np.median(ratio))
                p5, p95 = np.percentile(ratio, [5, 95])
                sweep = float((pred.max() - pred.min()) / np.median(pred))
                if (p95 - p5) / med > spread_tol or sweep < min_sweep:
                    break
                ks.append(round(med) if abs(med - round(med)) / med < integer_tol else med)
            else:
                for tr, k in zip(r.family.tracks, ks, strict=True):
                    tr.k = float(k)
                _join(a.family, r.family.tracks)
                refit = _fit_one(a.family, c)
                results[results.index(a)] = refit
                anchors[anchors.index(a)] = refit
                absorbed.add(idx)
                break
    return [r for i, r in enumerate(results) if i not in absorbed]


def fit_families(families: Sequence[Family], c: float = SPEED_OF_SOUND_15C) -> list[FamilyResult]:
    results = merge_consistent_families([_fit_one(fam, c) for fam in families], c)
    results = absorb_by_model(results, c)
    for i, r in enumerate(results):  # renumber after merges
        r.family.id = i
        for tr in r.family.tracks:
            tr.family_id = i
    return results


def analyze(
    x: NDArray,
    sample_rate: float,
    t0: float = 0.0,
    *,
    mutes: Sequence[tuple[float, float]] = (),
    noise_range: tuple[float, float] | None = None,
    spectro: SpectroSettings = SpectroSettings(),
    tracker: TrackerSettings = TrackerSettings(),
    highpass_hz: float | None = 15.0,
    c: float = SPEED_OF_SOUND_15C,
) -> AnalysisResult:
    """Run the full M2 chain on a mono signal whose first sample is at media time ``t0``.

    ``sigma`` passed to the fit is the per-peak Cramer-Rao estimate; the fit
    still rescales the covariance by the observed scatter (absolute_sigma=False).
    """
    if highpass_hz:
        x = highpass(x, sample_rate, highpass_hz)
    spec = compute_spectrogram(x, sample_rate, spectro, t0=t0, mutes=mutes)
    if noise_range is not None:
        spec = subtract_noise(spec, noise_range)
    white = spec.whitened_db()
    if tracker.stationary_window_s > 0:
        white = remove_stationary_lines(white, spectro.hop_s, tracker.stationary_window_s,
                                        valid=spec.frame_valid)
    if tracker.method == "ridge":
        raw = ridge_tracks(spec, white, tracker)
    else:
        peaks = pick_peaks(spec, tracker.threshold_db, tracker.max_peaks_per_frame, white=white)
        raw = link_tracks(peaks, spec.times, tracker)
    results = fit_families(group_families(raw, hop_s=spectro.hop_s), c)
    tracks = sorted((tr for r in results for tr in r.family.tracks), key=lambda tr: tr.id)
    return AnalysisResult(spec, tracks, results, t0)


def _family_time(fr: FamilyResult, w0: float, w1: float) -> float:
    """Where a family 'lives' for window ownership: its fitted CPA when that lies
    inside the window [w0, w1] (also when the window holds only a tail of the pass:
    that tail then defers to the window that holds the CPA), else the middle of
    its points (failed fits, or a CPA extrapolated far off)."""
    t0, t1 = fr.family.t_range
    if fr.fit is not None and fr.fit.success and w0 <= fr.fit.t_cpa <= w1:
        return float(fr.fit.t_cpa)
    return 0.5 * (t0 + t1)


DISPLAY_MAX_FRAMES = 12_000   # stitched overview spectrogram of a windowed analysis
DISPLAY_MAX_BINS = 2_500


def _window_plan(dur: float, window_s: float, overlap_s: float) -> tuple[list[float], list[float]]:
    """Window starts (s from the first sample) and the core boundaries between them
    (the middle of each overlap; -inf/inf at the ends)."""
    step = window_s - overlap_s
    starts = list(np.arange(0.0, max(dur - window_s, 0.0) + 1e-9, step))
    if starts[-1] + window_s < dur:
        starts.append(dur - window_s)
    bounds = [-np.inf] + [0.5 * (starts[i + 1] + starts[i] + window_s) for i in range(len(starts) - 1)] + [np.inf]
    return starts, bounds


def _pool_factors(dur: float, sp: SpectroSettings) -> tuple[int, int]:
    rt = max(1, int(np.ceil(dur / sp.hop_s / DISPLAY_MAX_FRAMES)))
    n_bins = (sp.fmax - sp.fmin) * sp.window_s * sp.zero_pad
    return rt, max(1, int(np.ceil(n_bins / DISPLAY_MAX_BINS)))


def _pool_core(spec: Spectrogram, lo: float, hi: float, rt: int, rf: int):
    """The frames of ``spec`` with time in [lo, hi), max-pooled by ``rt`` frames and
    ``rf`` bins (float32 power; a frame block is valid only if all its frames are)."""
    sel = np.flatnonzero((spec.times >= lo) & (spec.times < hi))
    nt = sel.size // rt
    nf = spec.freqs.size // rf
    if nt == 0 or nf == 0:
        return None
    sel = sel[: nt * rt]
    p = spec.power[: nf * rf][:, sel].astype(np.float32)
    p = p.reshape(nf, rf, nt, rt).max(axis=(1, 3))
    times = spec.times[sel].reshape(nt, rt).mean(1)
    freqs = spec.freqs[: nf * rf].reshape(nf, rf).mean(1)
    valid = spec.frame_valid[sel].reshape(nt, rt).all(1)
    return times, freqs, p, valid


def _analyze_window(x: NDArray, sample_rate: float, t_start: float, lo: float, hi: float, t0: float,
                    pool: tuple[int, int], kwargs: dict):
    """One window of ``analyze_windowed`` (top level so a worker process can run it):
    the families whose home lies in the core [lo, hi) (relative to ``t0``), the
    number found, and the core's pooled display spectrogram."""
    r = analyze(x, sample_rate, t_start, **kwargs)
    w0, w1 = t_start, t_start + x.size / sample_rate
    kept = [fr for fr in r.families if lo <= _family_time(fr, w0, w1) - t0 < hi]
    return kept, len(r.families), _pool_core(r.spectrogram, t0 + lo, t0 + hi, *pool), r.spectrogram.settings


def analyze_windowed(
    x: NDArray,
    sample_rate: float,
    t0: float = 0.0,
    *,
    window_s: float = 600.0,
    overlap_s: float = 150.0,
    workers: int | None = None,
    log=lambda m: None,
    progress=None,
    **kwargs,
) -> AnalysisResult:
    """``analyze`` over overlapping windows, for recordings too long for one
    spectrogram. A pass near a window edge is cut, so each window keeps only the
    families that live (CPA, else midpoint) in its core: the window minus half the
    overlap on each inner side. A pass is then fitted in the window that holds it
    best. Family and track ids are renumbered to stay unique. The returned
    spectrogram is a display overview of the whole recording: each window's core,
    max-pooled to at most ~DISPLAY_MAX_FRAMES x DISPLAY_MAX_BINS (a full-resolution
    one for 75 min would be ~1.8 GB); the tracking itself always ran at full
    resolution.

    ``workers`` > 1 runs windows in that many processes. Default 1: the ridge
    tracker's DP is memory-bandwidth bound (a 49 x 10k candidate matrix per frame),
    so on the dev machine 4 processes each ran ~4x slower (Manchester: 4 windows
    285 s serial, 304 s with 4 workers) for 4x the memory."""
    n = x.size
    dur = n / sample_rate
    if dur <= window_s:
        if progress:
            progress("tracking tones", 0, 0)
        return analyze(x, sample_rate, t0, **kwargs)
    starts, bounds = _window_plan(dur, window_s, overlap_s)
    sp = kwargs.get("spectro", SpectroSettings())
    rt, rf = _pool_factors(dur, sp)
    jobs = []
    for i, a in enumerate(starts):
        i0, i1 = int(round(a * sample_rate)), min(n, int(round((a + window_s) * sample_rate)))
        jobs.append((x[i0:i1], sample_rate, t0 + i0 / sample_rate, bounds[i], bounds[i + 1], t0, (rt, rf),
                     kwargs))
    if workers is None:
        workers = 1
    workers = max(1, min(workers, len(jobs)))
    if workers == 1:
        outs = map(lambda j: _analyze_window(*j), jobs)
    else:
        # forkserver: safe when the caller has threads (the GUI's worker pool).
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver"))
        outs = pool.map(_analyze_window, *zip(*jobs))
    families: list[FamilyResult] = []
    parts, settings = [], sp
    if progress:
        progress(f"tracking window 1/{len(jobs)}", 0, len(jobs))
    try:
        for i, (a, (kept, n_found, part, settings)) in enumerate(zip(starts, outs)):
            if progress:
                progress(f"tracking window {min(i + 2, len(jobs))}/{len(jobs)}", i + 1, len(jobs))
            if part is not None:
                parts.append(part)
            log(f"window {t0 + a:.0f}-{t0 + a + window_s:.0f} s: {n_found} families, {len(kept)} kept")
            families.extend(kept)
    finally:
        if workers > 1:
            pool.shutdown(cancel_futures=True)
    tid = 0
    for i, fr in enumerate(families):
        fr.family.id = i
        for tr in fr.family.tracks:
            tr.family_id, tr.id = i, tid
            tid += 1
    tracks = [tr for fr in families for tr in fr.family.tracks]
    overview = None
    if parts:
        overview = Spectrogram(
            np.concatenate([p[0] for p in parts]), parts[0][1], np.concatenate([p[2] for p in parts], axis=1),
            replace(settings, hop_s=settings.hop_s * rt, whiten_bins=max(5, (settings.whiten_bins // rf) | 1)),
            np.concatenate([p[3] for p in parts]))
    return AnalysisResult(overview, tracks, families, t0)


def analyze_file(
    path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE, *, window_s: float = 600.0,
    overlap_s: float = 150.0, workers: int | None = None, log=lambda m: None, progress=None, **kwargs
) -> AnalysisResult:
    """Load and analyse a media file; files longer than ``window_s`` are analysed
    in overlapping windows (``analyze_windowed``)."""
    if progress:
        progress("decoding audio", 0, 0)
    clip = load_audio(path, sample_rate)
    res = analyze_windowed(clip.samples, clip.sample_rate, clip.pts_offset, window_s=window_s,
                           overlap_s=overlap_s, workers=workers, log=log, progress=progress, **kwargs)
    res.clip = clip
    return res


def refit_families(tracks: Sequence[ToneTrack], c: float = SPEED_OF_SOUND_15C,
                   locks: dict[int, dict[str, float]] | None = None) -> list[FamilyResult]:
    """Fit the families exactly as the tracks are labelled (``family_id``), with
    no regrouping, merging or step trimming: the step after a user edits tracks
    (delete, move a track to another family, set its harmonic number, draw one)
    or restores a project. ``locks`` holds parameters per family at a value
    (``{family_id: {"v": 82.0}}``). Tracks without a family are left out."""
    by_fam: dict[int, list[ToneTrack]] = {}
    for tr in tracks:
        if tr.family_id is not None and tr.t.size:
            by_fam.setdefault(int(tr.family_id), []).append(tr)
    locks = locks or {}
    return [_fit_one(Family(fid, sorted(trs, key=lambda tr: tr.id)), c, trim=False, fixed=locks.get(fid))
            for fid, trs in sorted(by_fam.items())]


# ---------------------------------------------------------------------------
# Display spectrogram without tracking, and the analysis cache
# ---------------------------------------------------------------------------


def display_spectrogram(x: NDArray, sample_rate: float, t0: float = 0.0, *, window_s: float = 600.0,
                        overlap_s: float = 150.0, spectro: SpectroSettings = SpectroSettings(),
                        mutes: Sequence[tuple[float, float]] = (), noise_range: tuple[float, float] | None = None,
                        highpass_hz: float | None = 15.0, **_ignored) -> Spectrogram:
    """The spectrogram ``analyze_windowed`` would return, without the tracking: the
    full one for a short recording, else the pooled overview of the window cores.
    Used when the tracks come from the analysis cache."""
    if highpass_hz:
        x = highpass(x, sample_rate, highpass_hz)
    dur = x.size / sample_rate

    def one(seg: NDArray, ts: float) -> Spectrogram:
        spec = compute_spectrogram(seg, sample_rate, spectro, t0=ts, mutes=mutes)
        return subtract_noise(spec, noise_range) if noise_range is not None else spec

    if dur <= window_s:
        return one(x, t0)
    starts, bounds = _window_plan(dur, window_s, overlap_s)
    rt, rf = _pool_factors(dur, spectro)
    parts, settings = [], spectro
    for i, a in enumerate(starts):
        i0, i1 = int(round(a * sample_rate)), min(x.size, int(round((a + window_s) * sample_rate)))
        spec = one(x[i0:i1], t0 + i0 / sample_rate)
        settings = spec.settings
        part = _pool_core(spec, t0 + bounds[i], t0 + bounds[i + 1], rt, rf)
        if part is not None:
            parts.append(part)
    return Spectrogram(
        np.concatenate([p[0] for p in parts]), parts[0][1], np.concatenate([p[2] for p in parts], axis=1),
        replace(settings, hop_s=settings.hop_s * rt, whiten_bins=max(5, (settings.whiten_bins // rf) | 1)),
        np.concatenate([p[3] for p in parts]))


ANALYSIS_CACHE_VERSION = 1


def _code_fingerprint() -> str:
    """Hash of the analysis source files: a cached analysis made by older code is never reused."""
    import hashlib

    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("pipeline.py", "tracking.py", "spectral.py", "doppler.py", "media.py", "atmosphere.py"):
        try:
            h.update((here / name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:16]


def analysis_cache_key(media_sha256: str, **settings) -> str:
    """Cache key: media content + analysis settings (spectro, tracker, mutes, ...) + code."""
    import hashlib

    blob = repr((ANALYSIS_CACHE_VERSION, media_sha256, _code_fingerprint(), sorted(
        (k, repr(v)) for k, v in settings.items())))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def save_cached_analysis(folder: str | Path, key: str, res: AnalysisResult) -> Path:
    """Store the expensive part of an analysis (tracks, fitted families); the
    spectrogram and audio are rebuilt on load."""
    import pickle

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.pkl"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"tracks": res.tracks, "families": res.families, "t0": res.t0}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_cached_analysis(folder: str | Path, key: str) -> dict | None:
    import pickle

    path = Path(folder) / f"{key}.pkl"
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return None


def analyze_file_cached(path: str | Path, cache_folder: str | Path | None, media_sha256: str,
                        sample_rate: int = DEFAULT_SAMPLE_RATE, progress=None, log=lambda m: None,
                        **kwargs) -> tuple[AnalysisResult, bool]:
    """``analyze_file`` through the analysis cache: (result, came from the cache).
    A hit still decodes the audio (clip metadata, display spectrogram) but skips
    the tone tracking and fitting."""
    key = analysis_cache_key(media_sha256, sample_rate=sample_rate, **{k: v for k, v in kwargs.items()
                                                                    if k not in ("workers",)})
    hit = load_cached_analysis(cache_folder, key) if cache_folder else None
    if hit is not None:
        if progress:
            progress("decoding audio (analysis from cache)", 0, 0)
        clip = load_audio(path, sample_rate)
        if progress:
            progress("spectrogram", 0, 0)
        spec = display_spectrogram(clip.samples, clip.sample_rate, clip.pts_offset, **kwargs)
        res = AnalysisResult(spec, hit["tracks"], hit["families"], hit["t0"], clip)
        log(f"analysis loaded from cache ({len(res.families)} families)")
        return res, True
    res = analyze_file(path, sample_rate, progress=progress, log=log, **kwargs)
    if cache_folder:
        try:
            save_cached_analysis(cache_folder, key, res)
        except OSError as e:
            log(f"could not cache the analysis: {e}")
    return res, False
