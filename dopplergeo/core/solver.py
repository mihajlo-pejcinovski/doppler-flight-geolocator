"""Geolocation solver: gate candidates, grid search, joint refinement, mirror
ambiguity and grading (spec section 'Geolocation solver').

Frames and clocks
-----------------
* Everything is computed in one Cartesian ENU frame whose origin is the centre
  of the search region on the WGS84 ellipsoid. Aircraft and observers are
  converted into it exactly (no flat-earth approximation): an observer at
  (e, n) on the tangent plane is mapped to geodetic lat/lon, given its height,
  and mapped back.
* Times are seconds relative to the recording epoch (the file's UTC at media
  time 0), which keeps float64 resolution at nanoseconds instead of the 0.2 us
  of raw Unix epochs.
* The clock correction ``dt`` is added to the file clock:
  ``true UTC = epoch + t_media + dt``. Its bound is a hard limit ``|dt| <= B``.

Tracks are duck-typed: anything with ``icao24``, ``callsign``, ``t_start``,
``t_end``, ``position(t)``, ``velocity(t)`` (ECEF) and ``reliable(t)``, e.g.
``dopplergeo.flights.FlightTrack``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .atmosphere import SPEED_OF_SOUND_15C, Atmosphere, temperature_for_speed
from .doppler import DopplerFit, FloatArray
from .elevation import ConstantElevation, ElevationModel
from .forward import log_doppler_fast
from .geo import ecef_to_enu, enu_rotation, enu_to_geodetic, geodetic_to_enu

CHI2_2DOF_95 = 5.991464547107979
EARTH_R = 6_371_008.8


class SolverError(RuntimeError):
    pass


class Track(Protocol):
    icao24: str
    callsign: str | None

    @property
    def t_start(self) -> float: ...
    @property
    def t_end(self) -> float: ...
    def position(self, t: ArrayLike) -> FloatArray: ...
    def velocity(self, t: ArrayLike) -> FloatArray: ...
    def reliable(self, t: ArrayLike) -> NDArray[np.bool_]: ...


class Region(Protocol):
    def as_tuple(self) -> tuple[float, float, float, float]: ...


@dataclass
class SolverSettings:
    grid_step_m: float = 250.0
    clock_bound_s: float = 0.0          # hard bound on |dt|; 0 holds the file clock fixed
    # Gaussian prior on the clock offset, N(0, clock_sigma_s); 0 = flat within the bound.
    # Along a straight track position trades against time (1 s ~ 80 m on an approach),
    # so an original phone clock (NTP, ~1 s) pins the along-track position. It also
    # narrows the grid's clock search to +-4 sigma.
    clock_sigma_s: float = 0.0
    # ADS-B timing slack: each aircraft's track may be late or early by its own lag,
    # prior N(0, adsb_lag_sigma_s) (position latency of the transponder's GNSS, the
    # feeder's clock, engine vs antenna). Without it one shared clock must explain
    # every aircraft's timing exactly and the along-track error is overconfident.
    # 0 disables.
    adsb_lag_sigma_s: float = 0.5
    c: float = SPEED_OF_SOUND_15C
    # Observer height: an ElevationModel (DEM, ADS-B airport ground, ...) giving the
    # ellipsoidal height and its sigma per position; None -> constant observer_h_m.
    elevation: ElevationModel | None = None
    observer_h_m: float = 0.0
    observer_h_sigma_m: float = 100.0
    # Fit a height correction dh with the elevation model's sigma as prior. With one
    # aircraft the prior dominates but its uncertainty still reaches the ellipse; with
    # several look angles the data pin the height (+-1 m in tests).
    fit_height: bool = True
    fit_c: bool = False
    # Prior on the fitted surface speed of sound, N(s.c, c_sigma_mps); 0 = flat. With an
    # atmosphere the fitted c sets its temperature (1 m/s ~ 1.7 K). On approach passes the
    # tones never reach their asymptotes, so a 0.3 % error in c moves the position across
    # the track by ~300 m (Manchester): fitting c with an honest prior puts that in the ellipse.
    c_sigma_mps: float = 0.0
    # Air between observer and aircraft: temperature with the ISA lapse and a uniform
    # wind (e.g. Open-Meteo for the hour). None = still air at constant c (old model).
    # Wind along the line of sight changes the effective speed of sound by w/c (10 m/s
    # ~ 3 %), which biases the CPA distance and so the across-track position.
    atmos: Atmosphere | None = None
    # Measured ellipse: after the solve, hold the observer in ellipse_directions directions
    # and find where the fit worsens by the 95 % chi2 (5.99, in the covariance's units),
    # instead of trusting the Jacobian at the minimum. On Manchester the Jacobian ellipse
    # was 5-30x too confident along the approach (the clock trades ~1.2 s per 100 m and the
    # clipped chi2 valley flattens with distance). False keeps the Jacobian ellipse.
    measure_ellipse: bool = True
    ellipse_directions: int = 8
    ellipse_probe_m: float = 100.0      # first held position in each direction
    ellipse_max_m: float = 3000.0       # a direction still inside the 95 % level here is unbounded
    # Fit the wind (east, north) with the atmos wind as a Gaussian prior of this sigma.
    fit_wind: bool = False
    wind_sigma_mps: float = 3.0
    # gating (spec step 1)
    speed_tol: float = 0.25
    min_alt_frac: float = 0.3
    gate_margin_m: float = 5000.0
    gate_window_s: float = 60.0
    loose_speed_tol: float = 0.5        # for families whose fit is unreliable (FamilyObs.loose)
    loose_reach_m: float = 10_000.0
    # grid (spec step 2)
    grid_points: int = 48               # binned points per family used on the grid
    grid_res_factor: float = 0.3        # grid-resolution model error, see _grid_sigma
    cell_gate_factor: float = 1.5       # skip cells farther than 1.5 d + margin from the track
    cell_gate_margin_m: float = 1000.0
    min_valid_frac: float = 0.8
    common_clock: bool = True           # rescore the grid with one clock offset shared by all families
    chunk: int = 4096
    grid_workers: int = 0              # grid search threads; 0 = one per CPU core
    max_grid_cells: int = 600_000
    # refinement and ambiguity (spec steps 3-4)
    n_seeds: int = 3
    seed_separation_m: float = 1000.0
    mirror_dchi2: float = 9.0
    # robustness: a family that cannot be explained at a cell costs a fixed penalty
    # (a multiple of its own best achievable chi2) instead of vetoing the cell
    unexplained_factor: float = 25.0
    unexplained_min_chi2_per_bin: float = 4.0
    unexplained_max_chi2_per_bin: float = 400.0  # a family no cell explains better than this is ignored
    prune_factor: float = 10.0          # drop a family whose chi2_red exceeds 10x the median ...
    prune_floor: float = 3.0            # ... and this floor, after refinement
    max_chi2_red: float = 6.0           # hypotheses fitting worse than this (and 3x the best) rank last
    # Grade C when the joint fit is this many times worse (chi2_red) than the families'
    # own single-pass fits: real tones scatter beyond their per-peak sigma (single-pass
    # chi2_red 4-8 on good Manchester passes), so only the excess over that measures
    # disagreement between the passes and the position.
    max_chi2_excess: float = 3.0
    # Families graded A/B (and not loose) are solved alone when there are at least this
    # many: weak families with free clock offsets otherwise outvote them (Manchester, all
    # 52 families: 6.1 km off; its three grade-B families alone: 543 m). 0 disables.
    reliable_only_min: int = 2
    # Work caps for noisy real recordings (many weak families x many aircraft): pruning and
    # re-admission refine once per family/candidate, so the cost grows with their product.
    region_margin_m: float = 2000.0     # refined positions may leave the search region by this much
    max_families: int = 6               # best-graded families solved; the rest reported unexplained
    max_readmit_candidates: int = 8     # per left-out family, closest speed match first
    # aircraft the user ticked out (icao24): gated as usual, reported with included=False, never used
    exclude_aircraft: frozenset[str] = frozenset()
    # Robustness to tracking glitches (tone onsets, crossings): soft-L1 loss in the
    # refinement, and every chi2 used for ranking/pruning/grading clips each point at
    # outlier_clip sigma (a glitch costs at most 25, like a 5-sigma point).
    robust_f_scale: float = 3.0
    outlier_clip: float = 5.0
    node_dt_s: float = 0.5


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass
class FamilyObs:
    """One harmonic family's measured points (media time) and its single-pass fit."""

    id: int
    t: FloatArray
    f: FloatArray
    k: FloatArray
    sigma: FloatArray
    fit: DopplerFit

    @classmethod
    def from_result(cls, fr: Any) -> FamilyObs:
        """From a ``pipeline.FamilyResult`` with a fit."""
        if fr.fit is None:
            raise ValueError(f"family {fr.family.id} has no Doppler fit")
        t, f, k, s = fr.family.stacked()
        return cls(fr.family.id, t, f, k, s, fr.fit)

    loose_override: bool = False

    @property
    def loose(self) -> bool:
        """True when the single-pass fit cannot be trusted for gating: it failed,
        ran into a parameter bound (partial or accelerating/turning passes), or
        does not contain its own CPA. Such families are gated on their observed
        time span instead of v, d and t_cpa."""
        f = self.fit
        near_bound = not (40.0 < f.v < 300.0) or not (25.0 < f.d < 15_000.0)
        vague = f.grade == "C" and (f.stderr("v") / f.v > 0.10 or f.stderr("d") / f.d > 0.30)
        return self.loose_override or (not f.success) or near_bound or not f.cpa_covered or vague

    def emission_span(self, c: float, max_delay: float = 40.0) -> tuple[float, float]:
        return float(self.t.min()) - max_delay, float(self.t.max())

    def binned(self, n: int, sigma_extra: float = 0.0) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(t, log g, weight) with g = f/k averaged in ``n`` time bins; weights are
        1/sigma^2 of log g so the grid chi-square matches the full-data one.
        ``sigma_extra`` (Hz, in g units) adds model variance per bin."""
        g, sg = self.f / self.k, self.sigma / self.k
        w = 1.0 / sg**2
        edges = np.linspace(self.t.min(), self.t.max() + 1e-9, n + 1)
        idx = np.clip(np.searchsorted(edges, self.t, side="right") - 1, 0, n - 1)
        W = np.bincount(idx, w, n)
        keep = W > 0
        tb = np.bincount(idx, w * self.t, n)[keep] / W[keep]
        gb = np.bincount(idx, w * g, n)[keep] / W[keep]
        return tb, np.log(gb), gb**2 / (1.0 / W[keep] + sigma_extra**2)


# ---------------------------------------------------------------------------
# Trajectories sampled in the solver frame
# ---------------------------------------------------------------------------


class SampledTrajectory:
    """A track resampled on uniform nodes in the solver ENU frame, evaluated by
    vectorised cubic Hermite interpolation; linear extrapolation past the ends
    (flagged invalid) keeps Newton iterations finite."""

    def __init__(self, track: Track, lat0: float, lon0: float, t_ref: float, dt: float = 0.5):
        t0, t1 = float(track.t_start), float(track.t_end)
        n = max(1, int(math.ceil((t1 - t0) / dt)))
        nodes = np.linspace(t0, t1, n + 1)
        self.track = track
        self.icao24 = track.icao24
        self.callsign = track.callsign
        self.t0 = t0 - t_ref
        self.t1 = t1 - t_ref
        self.dt = (t1 - t0) / n
        self.P = ecef_to_enu(track.position(nodes), lat0, lon0, 0.0)
        self.V = np.asarray(track.velocity(nodes)) @ enu_rotation(lat0, lon0).T
        self.ok = np.asarray(track.reliable(nodes), dtype=bool) & np.isfinite(self.P).all(1)
        self.P = np.nan_to_num(self.P)
        self.V = np.nan_to_num(self.V)

    def eval(self, tau: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        x = (np.asarray(tau, dtype=float) - self.t0) / self.dt
        last = self.P.shape[0] - 2
        i = np.clip(np.floor(x), 0, last).astype(np.intp)
        u = x - i
        uc = np.clip(u, 0.0, 1.0)[..., None]
        u2, u3 = uc * uc, uc * uc * uc
        P0, P1, V0, V1 = self.P[i], self.P[i + 1], self.V[i] * self.dt, self.V[i + 1] * self.dt
        pos = (2 * u3 - 3 * u2 + 1) * P0 + (u3 - 2 * u2 + uc) * V0 + (3 * u2 - 2 * u3) * P1 + (u3 - u2) * V1
        vel = ((6 * u2 - 6 * uc) * P0 + (3 * u2 - 4 * uc + 1) * V0 + (6 * uc - 6 * u2) * P1
               + (3 * u2 - 2 * uc) * V1) / self.dt
        beyond = (u - uc[..., 0]) * self.dt
        pos = pos + beyond[..., None] * vel
        valid = (u >= -1e-9) & (u <= 1 + 1e-9) & self.ok[i] & self.ok[i + 1]
        return pos, vel, valid

    def samples(self, t0: float, t1: float, step: float = 1.0) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Valid (t, P, V) on a grid inside [t0, t1] ∩ the track span."""
        a, b = max(t0, self.t0), min(t1, self.t1)
        if b <= a:
            return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
        t = np.arange(a, b + 1e-9, step)
        P, V, ok = self.eval(t)
        return t[ok], P[ok], V[ok]


def log_doppler(traj: SampledTrajectory, obs: FloatArray, t: FloatArray, c: float,
                iters: int | None = None, atmos: Atmosphere | None = None, parallel: bool | None = None
                ) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """log of the Doppler factor 1/(1 + v_r/c) for reception times ``t`` at
    observer(s) ``obs`` (broadcast to ``t.shape + (3,)``). Returns (logD, valid, tau).
    With ``atmos`` the air has a height-dependent speed of sound and a uniform wind
    (``_log_doppler_air``); ``c`` is then unused. Runs the compiled kernel in
    ``forward``; ``log_doppler_numpy`` is the reference implementation."""
    return log_doppler_fast(traj, obs, t, c, atmos=atmos, iters=iters, parallel=parallel)


def log_doppler_numpy(traj: SampledTrajectory, obs: FloatArray, t: FloatArray, c: float,
                      iters: int = 10, atmos: Atmosphere | None = None
                      ) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """NumPy reference of ``log_doppler`` (same model, same arguments)."""
    if atmos is not None:
        return _log_doppler_air(traj, obs, t, atmos, iters)
    p, _, _ = traj.eval(t)
    tau = t - np.linalg.norm(p - obs, axis=-1) / c
    for _ in range(iters):
        p, v, _ = traj.eval(tau)
        r = p - obs
        dist = np.linalg.norm(r, axis=-1)
        vr = np.einsum("...i,...i->...", v, r) / dist
        step = (tau + dist / c - t) / (1.0 + vr / c)
        tau = tau - step
        if np.max(np.abs(step), initial=0.0) < 1e-8:
            break
    p, v, valid = traj.eval(tau)
    r = p - obs
    vr = np.einsum("...i,...i->...", v, r) / np.linalg.norm(r, axis=-1)
    # Clip only matters for extrapolated (invalid) points, which are masked anyway.
    return -np.log1p(np.clip(vr / c, -0.95, 0.95)), valid & np.isfinite(vr), tau


def _log_doppler_air(traj: SampledTrajectory, obs: FloatArray, t: FloatArray, atmos: Atmosphere,
                     iters: int = 12) -> tuple[FloatArray, NDArray[np.bool_], FloatArray]:
    """Doppler factor in moving air, exact for a uniform wind w.

    In the air's frame a wavefront emitted at tau from p(tau) is a sphere of radius
    c (t - tau) whose centre drifts with the wind, so it reaches the observer when
    |q| = c (t - tau), q = obs - p(tau) - w (t - tau). Differentiating that in t
    gives the received/emitted frequency ratio

        f / f0 = dtau/dt = (c + q^.w) / (c + q^.(w - v)),

    which reduces to 1 / (1 + v_r / c) in still air. c is the travel-time mean
    speed of sound between the observer's and the aircraft's heights (ISA lapse),
    re-evaluated as the emission point moves. Newton on g(tau) = |q| - c (t - tau)
    with g' = c + q^.(w - v) (the slow change of c with tau is left out of g'; it
    only slows convergence slightly)."""
    w = atmos.wind
    p, _, _ = traj.eval(t)
    c = atmos.c_path(p[..., 2] - obs[..., 2])
    tau = t - np.linalg.norm(p - obs, axis=-1) / c
    for _ in range(iters):
        p, v, _ = traj.eval(tau)
        c = atmos.c_path(p[..., 2] - obs[..., 2])
        s = t - tau
        q = obs - p - w * s[..., None]
        qn = np.linalg.norm(q, axis=-1)
        qh = q / np.maximum(qn, 1e-9)[..., None]
        gp = c + np.einsum("...i,...i->...", qh, w - v)
        step = (qn - c * s) / np.maximum(gp, 0.05 * c)
        tau = tau - step
        if np.max(np.abs(step), initial=0.0) < 1e-8:
            break
    p, v, valid = traj.eval(tau)
    c = atmos.c_path(p[..., 2] - obs[..., 2])
    q = obs - p - w * (t - tau)[..., None]
    qh = q / np.maximum(np.linalg.norm(q, axis=-1), 1e-9)[..., None]
    num = c + qh @ w
    den = c + np.einsum("...i,...i->...", qh, w - v)
    ok = valid & np.isfinite(den)
    return np.log(num / np.clip(den, 0.05 * c, None)), ok, tau


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    lat0: float
    lon0: float
    epoch: float  # file UTC at media time 0

    def observer(self, e: ArrayLike, n: ArrayLike, h: ArrayLike) -> FloatArray:
        """Tangent-plane (e, n) + ellipsoidal height -> exact Cartesian ENU."""
        e, n = np.asarray(e, dtype=float), np.asarray(n, dtype=float)
        lat, lon, _ = enu_to_geodetic(np.stack([e, n, np.zeros_like(e)], axis=-1), self.lat0, self.lon0)
        return geodetic_to_enu(lat, lon, np.broadcast_to(h, lat.shape), self.lat0, self.lon0)

    def latlon(self, e: ArrayLike, n: ArrayLike) -> tuple[FloatArray, FloatArray]:
        e, n = np.asarray(e, dtype=float), np.asarray(n, dtype=float)
        lat, lon, _ = enu_to_geodetic(np.stack([e, n, np.zeros_like(e)], axis=-1), self.lat0, self.lon0)
        return lat, lon

    def en(self, lat: float, lon: float) -> tuple[float, float]:
        p = geodetic_to_enu(lat, lon, 0.0, self.lat0, self.lon0)
        return float(p[0]), float(p[1])


def elevation_model(s: SolverSettings) -> ElevationModel:
    return s.elevation if s.elevation is not None else \
        ConstantElevation(s.observer_h_m, s.observer_h_sigma_m, "constant")


def observer_height(frame: Frame, e: ArrayLike, n: ArrayLike, s: SolverSettings) -> tuple[FloatArray, FloatArray]:
    """Observer ellipsoidal height and 1-sigma at tangent-plane positions."""
    lat, lon = frame.latlon(e, n)
    return elevation_model(s).height(lat, lon)


def _region_en(frame: Frame, region: Region) -> tuple[float, float, float, float]:
    lat_min, lon_min, lat_max, lon_max = region.as_tuple()
    lats = np.array([lat_min, lat_min, lat_max, lat_max, lat_min, lat_max, frame.lat0, frame.lat0])
    lons = np.array([lon_min, lon_max, lon_min, lon_max, frame.lon0, frame.lon0, lon_min, lon_max])
    p = geodetic_to_enu(lats, lons, 0.0, frame.lat0, frame.lon0)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


# ---------------------------------------------------------------------------
# Step 1: gating
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    family_id: int
    icao24: str
    callsign: str | None
    traj: SampledTrajectory = field(repr=False)
    t_cpa_utc: float           # when the track passes closest to the region centre
    cpa_range_m: float         # horizontal distance of that pass from the centre
    speed_match: float         # ADS-B ground speed / fitted v
    included: bool = True
    chi2: float = float("nan")  # at the final solution (f0 re-fitted)


def gate_candidates(families: Sequence[FamilyObs], tracks: Sequence[Track | SampledTrajectory],
                    frame: Frame, region: Region, s: SolverSettings) -> list[Candidate]:
    trajs = [t if isinstance(t, SampledTrajectory) else
             SampledTrajectory(t, frame.lat0, frame.lon0, frame.epoch, s.node_dt_s) for t in tracks]
    e0, e1, n0, n1 = _region_en(frame, region)
    out: list[Candidate] = []
    h_ref = float(observer_height(frame, 0.0, 0.0, s)[0])
    for fam in families:
        fam.loose_override = False
        found = _gate_family(fam, fam.loose, trajs, frame, (e0, e1, n0, n1), s, h_ref)
        if not found and not fam.loose:
            fam.loose_override = True  # strict gates found nothing: trust the time span only
            found = _gate_family(fam, True, trajs, frame, (e0, e1, n0, n1), s, h_ref)
        out += found
    return out


def _gate_family(fam: FamilyObs, loose: bool, trajs: Sequence[SampledTrajectory], frame: Frame,
                 box: tuple[float, float, float, float], s: SolverSettings, h_ref: float) -> list[Candidate]:
    e0, e1, n0, n1 = box
    out: list[Candidate] = []
    fit = fam.fit
    if loose:
        a, b = fam.emission_span(s.c)
        win = (a - s.clock_bound_s, b + s.clock_bound_s)
        v_ok = 40.0 < fit.v < 300.0
        reach = s.loose_reach_m
    else:
        tau_c = fit.t_cpa - fit.d / s.c  # emission time of CPA, media seconds
        w = s.gate_window_s + s.clock_bound_s
        win = (tau_c - w, tau_c + w)
        reach = 1.5 * fit.d + s.gate_margin_m
    for tr in trajs:
        t, P, V = tr.samples(*win)
        if t.size < 5:
            continue
        gs = float(np.median(np.hypot(V[:, 0], V[:, 1])))
        match = gs / fit.v
        if not loose and abs(match - 1.0) > s.speed_tol:
            continue
        if loose and v_ok and abs(match - 1.0) > s.loose_speed_tol:
            continue
        dx = np.maximum.reduce([e0 - P[:, 0], np.zeros(len(P)), P[:, 0] - e1])
        dy = np.maximum.reduce([n0 - P[:, 1], np.zeros(len(P)), P[:, 1] - n1])
        if np.hypot(dx, dy).min() > reach:
            continue
        alt = P[:, 2] + (P[:, 0] ** 2 + P[:, 1] ** 2) / (2 * EARTH_R) - h_ref
        if not loose and np.median(alt) < s.min_alt_frac * fit.d:
            continue
        hd = np.hypot(P[:, 0], P[:, 1])
        j = int(np.argmin(hd))
        out.append(Candidate(fam.id, tr.icao24, tr.callsign, tr, frame.epoch + float(t[j]),
                             float(hd[j]), match))
    return out


# ---------------------------------------------------------------------------
# Step 2: grid search
# ---------------------------------------------------------------------------


@dataclass
class GridResult:
    e: FloatArray                 # (nx,) tangent-plane east of cell centres
    n: FloatArray                 # (ny,)
    chi2: FloatArray              # (ny, nx) total chi-square, inf where unexplained
    family_chi2: FloatArray       # (F, ny, nx)
    assign: NDArray[np.intp]      # (F, ny, nx) candidate index per family, -1 none
    dt: FloatArray                # (F, ny, nx) clock correction used per family
    family_ids: list[int]
    penalty: FloatArray = field(default_factory=lambda: np.empty(0))  # (F,) unexplained cost
    lat: FloatArray = field(repr=False, default_factory=lambda: np.empty(0))
    lon: FloatArray = field(repr=False, default_factory=lambda: np.empty(0))

    def heatmap(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(lat, lon, delta chi2 above the minimum); NaN where nothing fits."""
        d = self.chi2 - np.nanmin(np.where(np.isfinite(self.chi2), self.chi2, np.nan))
        return self.lat, self.lon, np.where(np.isfinite(d), d, np.nan)


def _grid_sigma(fam: FamilyObs, step: float, s: SolverSettings) -> float:
    """Frequency error (Hz, k=1) from the observer being up to half a cell off:
    roughly the change of the S-curve near CPA when d changes by step/2. Without
    it, sharp (low-noise) data make the grid score depend on how close each cell
    happens to fall to the valley rather than on which hypothesis is right."""
    fit = fam.fit
    v = float(np.clip(fit.v, 80.0, 250.0)) if fam.loose else fit.v
    d = float(np.clip(fit.d, 500.0, 8000.0)) if fam.loose else max(fit.d, 100.0)
    f0 = float(np.median(fam.f / fam.k)) if fam.loose else fit.f0
    return s.grid_res_factor * f0 * (v / s.c) * (0.5 * step) / d


def grid_search(families: Sequence[FamilyObs], cands: Sequence[Candidate], frame: Frame,
                region: Region, s: SolverSettings, progress=None) -> GridResult:
    e0, e1, n0, n1 = _region_en(frame, region)
    step = s.grid_step_m
    while ((e1 - e0) / step + 1) * ((n1 - n0) / step + 1) > s.max_grid_cells:
        step *= 1.25
    ge = np.arange(e0, e1 + 1e-6, step)
    gn = np.arange(n0, n1 + 1e-6, step)
    E, N = np.meshgrid(ge, gn)
    E, N = E.ravel(), N.ravel()
    obs_all = frame.observer(E, N, observer_height(frame, E, N, s)[0])  # terrain per cell
    F, M = len(families), E.size
    B = s.clock_bound_s
    prep = []
    for fam in families:
        tb, lg, wl = fam.binned(s.grid_points, _grid_sigma(fam, step, s))
        d = fam.fit.d
        tau_c = fam.fit.t_cpa - d / s.c
        if fam.loose:
            a_, b_ = fam.emission_span(s.c)
            span, reach = (a_ - B, b_ + B), s.loose_reach_m
        else:
            span = (tau_c - s.gate_window_s - B - d / s.c, tau_c + s.gate_window_s + B + d / s.c)
            reach = s.cell_gate_factor * d + s.cell_gate_margin_m
        prep.append((tb, lg, wl, span, reach))

    two_pass = s.common_clock and s.clock_bound_s > 0 and len(families) >= 2 \
        and any(not f.loose for f in families)
    nE = ge.size
    jobs = [(fi, ci) for fi in range(F) for ci, cand in enumerate(cands) if cand.family_id == families[fi].id]
    workers = max(1, min(s.grid_workers or (os.cpu_count() or 1), len(jobs)))
    n_steps = len(jobs) * (2 if two_pass else 1)
    step_no = [0]

    def near_cells(SP: FloatArray, reach: float) -> NDArray[np.intp]:
        """Grid cells within ``reach`` (horizontally) of the track samples ``SP``.
        Only the block of the regular grid around the track's bounding box is
        tested, so the cost follows the track's size, not the region's."""
        ia = int(np.searchsorted(ge, SP[:, 0].min() - reach, side="left"))
        ib = int(np.searchsorted(ge, SP[:, 0].max() + reach, side="right"))
        ja = int(np.searchsorted(gn, SP[:, 1].min() - reach, side="left"))
        jb = int(np.searchsorted(gn, SP[:, 1].max() + reach, side="right"))
        if ia >= ib or ja >= jb:
            return np.empty(0, dtype=np.intp)
        block = (np.arange(ja, jb)[:, None] * nE + np.arange(ia, ib)[None, :]).ravel()
        keep = []
        for k in range(0, block.size, s.chunk):
            c = block[k:k + s.chunk]
            hd = np.hypot(E[c, None] - SP[None, :, 0], N[c, None] - SP[None, :, 1]).min(1)
            keep.append(c[hd <= reach])
        return np.concatenate(keep) if keep else np.empty(0, dtype=np.intp)

    def score(fi: int, ci: int, fixed_dt: FloatArray | None, par: bool | None):
        """One (family, candidate): chi2 and clock offset at the cells it reaches."""
        fam, cand = families[fi], cands[ci]
        tb, lg, wl, (span_lo, span_hi), reach = prep[fi]
        wtot = wl.sum()
        st, SP, _ = cand.traj.samples(span_lo, span_hi)
        if st.size < 2:
            return None
        near = near_cells(SP, reach)
        if fixed_dt is not None:
            near = near[np.isfinite(fixed_dt[near])]
        out_c, out_chi, out_dt = [], [], []
        for a in range(0, near.size, s.chunk):
            cells = near[a:a + s.chunk]
            obs = obs_all[cells]
            if fixed_dt is not None:
                dt = fixed_dt[cells]
            elif B > 0 and not fam.loose:
                d2 = ((obs[:, None, :] - SP[None, :, :]) ** 2).sum(-1)
                j = np.clip(d2.argmin(1), 1, st.size - 2)
                rows = np.arange(len(j))
                # Parabolic vertex of squared distance (exact for straight flight):
                # the 1 s sampling alone would quantise the CPA time to ±0.5 s.
                ym, y0, yp = d2[rows, j - 1], d2[rows, j], d2[rows, j + 1]
                curv = ym - 2 * y0 + yp
                off = np.where(curv > 0, 0.5 * (ym - yp) / np.where(curv > 0, curv, 1.0), 0.0)
                off = np.clip(off, -1.0, 1.0)
                h = st[j + 1] - st[j]
                tc = st[j] + off * h
                dmin2 = np.maximum(y0 - 0.25 * (ym - yp) * off, 0.0)
                t_pred = tc + np.sqrt(dmin2) / s.c
                dt = np.clip(t_pred - fam.fit.t_cpa, -B, B)
            else:
                dt = np.zeros(cells.size)  # loose families: clock comes from the others
            tt = tb[None, :] + dt[:, None]
            logD, valid, _ = log_doppler(cand.traj, obs[:, None, :], tt, s.c, atmos=s.atmos, parallel=par)
            w = wl[None, :] * valid
            wsum = w.sum(1)
            y = np.where(valid, lg[None, :] - logD, 0.0)
            lf0 = (w * y).sum(1) / np.maximum(wsum, 1e-300)
            chi = (w * (y - lf0[:, None]) ** 2).sum(1) * wtot / np.maximum(wsum, 1e-300)
            chi[wsum < s.min_valid_frac * wtot] = np.inf
            out_c.append(cells), out_chi.append(chi), out_dt.append(dt)
        if not out_c:
            return None
        return np.concatenate(out_c), np.concatenate(out_chi), np.concatenate(out_dt)

    def scan(fixed_dt: FloatArray | None, which: Sequence[int]):
        """Each family's best (chi2, candidate, dt) per cell. ``fixed_dt`` (per cell)
        imposes one clock offset on every family; otherwise strict families take
        the offset that puts their own CPA right and loose ones dt = 0.
        The (family, candidate) pairs run on ``workers`` threads (NumPy and the
        serial numba kernel release the GIL); results merge in pair order, so the
        outcome is the same as a serial scan."""
        fchi = np.full((F, M), np.inf)
        fasg = np.full((F, M), -1, dtype=np.intp)
        fdt = np.full((F, M), np.nan)
        which = set(which)
        todo = [(fi, ci) for fi, ci in jobs if fi in which]
        n_done = [0]

        def merge(fi: int, ci: int, r) -> None:
            n_done[0] += 1
            if progress:
                progress("grid search", step_no[0] + n_done[0], n_steps)
            if r is None:
                return
            cells, chi, dt = r
            better = chi < fchi[fi, cells]
            fchi[fi, cells[better]] = chi[better]
            fasg[fi, cells[better]] = ci
            fdt[fi, cells[better]] = dt[better]

        if workers == 1:
            for fi, ci in todo:
                merge(fi, ci, score(fi, ci, fixed_dt, None))
        else:
            ex = ThreadPoolExecutor(max_workers=workers)
            try:
                futs = [ex.submit(score, fi, ci, fixed_dt, False) for fi, ci in todo]
                for (fi, ci), fu in zip(todo, futs):
                    merge(fi, ci, fu.result())  # progress() may raise (a cancelled GUI job)
            except BaseException:
                # Drop the queued pairs instead of finishing them: a cancel takes effect
                # as soon as the running pairs (seconds) are done.
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            ex.shutdown()
        step_no[0] += len(todo)
        return fchi, fasg, fdt

    fchi, fasg, fdt = scan(None, range(F))
    strict = [fi for fi, f in enumerate(families) if not f.loose]
    if s.common_clock and B > 0 and len(families) >= 2 and strict:
        # One clock for the whole recording: at each cell take the median of the
        # offsets the strict families want (each judged by its own best aircraft),
        # then rescore every family at that common offset. With one free offset per
        # family a single straight pass slides along its track (dt x v); several
        # passes only agree on the offset at the right place.
        own = np.where(np.isfinite(fchi[strict]), fdt[strict], np.nan)
        has = np.isfinite(own).any(0)
        common = np.full(M, np.nan)
        common[has] = np.nanmedian(own[:, has], axis=0)
        fchi2, fasg2, fdt2 = scan(common, range(F))
        # Strict families: rescored where at least two of them set the common clock
        # (one alone already had its own best offset). Loose ones: wherever a strict
        # family sets it, instead of dt = 0.
        multi = np.isfinite(own).sum(0) >= 2
        for fi in range(F):
            use = multi if fi in strict else has
            fchi[fi, use], fasg[fi, use], fdt[fi, use] = fchi2[fi, use], fasg2[fi, use], fdt2[fi, use]
    # Unexplained-family penalty, relative to what each family can achieve at best.
    nbins = np.array([len(f.binned(s.grid_points)[0]) for f in families], dtype=float)
    floor = np.array([np.min(fc[np.isfinite(fc)]) if np.isfinite(fc).any() else np.inf for fc in fchi])
    penalty = np.clip(s.unexplained_factor * np.where(np.isfinite(floor), floor, 0.0),
                      s.unexplained_min_chi2_per_bin * nbins, s.unexplained_max_chi2_per_bin * nbins)
    unexpl = fchi >= penalty[:, None]
    fasg[unexpl] = -1
    total = np.where(unexpl, penalty[:, None], fchi).sum(0)
    total[unexpl.all(0)] = np.inf  # a cell must explain at least one family
    shape = (gn.size, ge.size)
    lat, lon = frame.latlon(E, N)
    return GridResult(ge, gn, total.reshape(shape), fchi.reshape((F,) + shape), fasg.reshape((F,) + shape),
                      fdt.reshape((F,) + shape), [f.id for f in families], penalty,
                      lat.reshape(shape), lon.reshape(shape))


def _grid_seeds(g: GridResult, n_seeds: int, sep: float) -> list[tuple[float, float, int]]:
    """Up to ``n_seeds`` lowest cells at least ``sep`` metres apart: (e, n, flat index)."""
    flat = g.chi2.ravel()
    order = np.argsort(flat)
    E, N = np.meshgrid(g.e, g.n)
    E, N = E.ravel(), N.ravel()
    seeds: list[tuple[float, float, int]] = []
    for i in order:
        if not np.isfinite(flat[i]) or len(seeds) >= n_seeds:
            break
        if all(math.hypot(E[i] - e, N[i] - n) >= sep for e, n, _ in seeds):
            seeds.append((float(E[i]), float(N[i]), int(i)))
    return seeds


# ---------------------------------------------------------------------------
# Step 3: joint refinement
# ---------------------------------------------------------------------------


@dataclass
class RefinedFit:
    e: float
    n: float
    h: float
    dt: float
    c: float
    lat: float
    lon: float
    chi2: float
    dof: int
    cov_en: FloatArray           # 2x2 (east, north) m^2, scaled by max(chi2_red, 1)
    assignment: dict[int, Candidate]
    f0: dict[int, float]
    success: bool
    message: str
    seed: str = "grid"
    family_chi2: dict[int, tuple[float, int]] = field(default_factory=dict)  # id -> (chi2, n points)
    h_prior: float = float("nan")      # height the elevation model gives at (e, n)
    h_sigma: float = float("nan")      # prior sigma, or the posterior one when the height was fitted
    h_fitted: bool = False
    corr_inflation: float = 1.0        # covariance factor for serially correlated residuals
    # Correlation-weighted chi2 and the covariance's scale (max(chi2_eff / dof_eff, 1)).
    # Between fits of the same families, (chi2_eff - chi2_eff_min) / scale_at_min is the
    # rise in sigma^2 units that the covariance (the ellipse) predicts as (offset / sigma)^2.
    chi2_eff: float = float("nan")
    chi2_scale: float = float("nan")
    lags: dict[str, float] = field(default_factory=dict)  # icao24 -> ADS-B time lag (s)
    atmos: Atmosphere | None = None    # the air used (wind fitted when fit_wind); None = still air

    def time_shift(self, icao24: str) -> float:
        """Media time -> track time offset for one aircraft: clock offset + its lag."""
        return self.dt + self.lags.get(icao24, 0.0)

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def ellipse_95(self) -> tuple[float, float, float]:
        """(semi-major m, semi-minor m, major-axis bearing deg clockwise from north)."""
        if not np.isfinite(self.cov_en).all():
            return float("inf"), float("inf"), 0.0
        lam, vec = np.linalg.eigh(self.cov_en)
        lam = np.clip(lam, 0.0, None)
        major = vec[:, 1]
        bearing = math.degrees(math.atan2(major[0], major[1])) % 180.0
        return math.sqrt(CHI2_2DOF_95 * lam[1]), math.sqrt(CHI2_2DOF_95 * lam[0]), bearing


MIN_VALID_POINTS = 10


def refine(families: Sequence[FamilyObs], assignment: dict[int, Candidate], frame: Frame,
           e0: float, n0: float, dt0: float, s: SolverSettings, seed: str = "grid",
           hold_position: bool = False) -> RefinedFit:
    """Joint least squares over (e, n[, dt][, dh][, c], log f0 per family[, drift]).

    The observer height is the elevation model's value at (e, n) plus ``dh``,
    which is fitted (``s.fit_height``) with the model's sigma as a Gaussian prior
    (one extra residual dh / sigma), so its uncertainty reaches the position
    covariance. Fitting it inside every hypothesis matters: with a wrong fixed
    height the passes disagree and pruning would throw away the aircraft needed."""
    fams = [f for f in families if f.id in assignment]
    B = s.clock_bound_s
    fit_h = bool(s.fit_height)
    # A family whose aircraft is out of range at the seed for (almost) all of its points
    # adds a rest-frequency parameter nothing constrains: J gets a zero column and the
    # covariance is singular (seen on Manchester: an infinite ellipse, graded C). Leave it out.
    h_seed = float(observer_height(frame, e0, n0, s)[0])
    obs_seed = frame.observer(e0, n0, h_seed)
    keep = []
    for f in fams:
        _, valid, _ = log_doppler(assignment[f.id].traj, obs_seed, f.t + dt0, s.c, atmos=s.atmos)
        if valid.sum() >= max(MIN_VALID_POINTS, s.min_valid_frac * 0.5 * valid.size):
            keep.append(f)
    if len(keep) < len(fams):
        assignment = {f.id: assignment[f.id] for f in keep}
        fams = keep
    if not fams:
        raise SolverError("no family has enough points in range of its aircraft")
    names = ["e", "n"] + (["dt"] if B > 0 else []) + (["dh"] if fit_h else []) \
        + (["c"] if s.fit_c else [])
    lag_sig = float(s.adsb_lag_sigma_s)
    aircraft = sorted({assignment[f.id].icao24 for f in fams}) if lag_sig > 0 else []
    names += [f"lag_{a}" for a in aircraft]
    fit_w = bool(s.fit_wind and s.atmos is not None)
    names += ["we", "wn"] if fit_w else []
    clock_sig = float(s.clock_sigma_s) if B > 0 else 0.0
    drift = {f.id: "f0_rate" in f.fit.param_names for f in fams}
    for f in fams:
        names.append(f"lf0_{f.id}")
        if drift[f.id]:
            names.append(f"rate_{f.id}")
    ix = {nm: i for i, nm in enumerate(names)}

    def air(x: FloatArray) -> Atmosphere | None:
        """The atmosphere at parameters x (fitted wind / speed of sound), or None."""
        if s.atmos is None:
            return None
        a = s.atmos
        if fit_w:
            a = replace(a, wind_e=float(x[ix["we"]]), wind_n=float(x[ix["wn"]]))
        if "c" in ix:
            a = replace(a, temp_c=temperature_for_speed(float(x[ix["c"]])))
        return a

    def unpack(x: FloatArray) -> tuple[float, float, float, float, float]:
        h = float(observer_height(frame, x[0], x[1], s)[0]) + (x[ix["dh"]] if "dh" in ix else 0.0)
        return (x[0], x[1], x[ix["dt"]] if "dt" in ix else 0.0, h, x[ix["c"]] if "c" in ix else s.c)

    h0, h_sig0 = (float(v) for v in observer_height(frame, e0, n0, s))
    # Valid mask fixed at the seed so the residual vector keeps its meaning.
    obs0 = frame.observer(e0, n0, h0)
    masks, x0 = {}, [e0, n0] + ([np.clip(dt0, -B, B)] if B > 0 else []) \
        + ([0.0] if fit_h else []) + ([s.c] if s.fit_c else []) + [0.0] * len(aircraft) \
        + ([s.atmos.wind_e, s.atmos.wind_n] if fit_w else [])
    for f in fams:
        logD, valid, _ = log_doppler(assignment[f.id].traj, obs0, f.t + dt0, s.c, atmos=s.atmos)
        masks[f.id] = valid
        w = (f.f / f.sigma) ** 2 * valid
        lf0 = float(np.sum(w * (np.log(f.f / f.k) - logD)) / max(w.sum(), 1e-300)) if valid.any() \
            else math.log(f.fit.f0)
        x0.append(lf0)
        if drift[f.id]:
            x0.append(0.0)
    x0 = np.asarray(x0, dtype=float)
    n_pts = int(sum(m.sum() for m in masks.values()))
    if n_pts <= len(names):
        raise SolverError("too few valid points to refine")
    # Weight of the clock and lag priors against the data. The audio residuals are
    # serially correlated (n_pts / n_eff points per independent one) and scatter beyond
    # their sigma (the passes' own chi2_red), so a plain (x/sigma)^2 prior would be
    # outvoted by thousands of points that carry far less information. Scaling it by
    # k makes the prior count as 1/sigma^2 against the data's effective information.
    n_eff0 = float(sum(m.sum() / max(f.fit.corr_factor, 1.0) for f, m in ((f, masks[f.id]) for f in fams)))
    own_red0 = sum(f.fit.chi2 for f in fams) / max(sum(f.fit.dof for f in fams), 1)
    k_prior = (n_pts / max(n_eff0, 1.0)) * max(own_red0, 1.0)
    n_priors = (1 if fit_h else 0) + len(aircraft) + (1 if clock_sig > 0 else 0)

    def data_resid(x: FloatArray) -> FloatArray:
        e, n, dt, h, c = unpack(x)
        atm = air(x)
        obs = frame.observer(e, n, h)
        out = []
        for f in fams:
            lag = x[ix[f"lag_{assignment[f.id].icao24}"]] if aircraft else 0.0
            logD, _, tau = log_doppler(assignment[f.id].traj, obs, f.t + dt + lag, c, atmos=atm)
            f0 = math.exp(x[ix[f"lf0_{f.id}"]])
            model = f.k * f0 * np.exp(logD)
            if drift[f.id]:
                model = model * (1.0 + x[ix[f"rate_{f.id}"]] * (tau - f.fit.t_cpa))
            r = (model - f.f) / f.sigma
            out.append(np.where(masks[f.id], r, 0.0))
        return np.concatenate(out)

    # Gaussian priors as (parameter index, 1/sigma including the k weight). Each row's
    # residual is x[i] * inv_sig; its cost is that squared (made exact under soft_l1
    # by _quadratic_under_soft_l1).
    priors: list[tuple[int, float, float]] = []      # (index, inv_sigma, covariance row weight)
    if fit_h:
        priors.append((ix["dh"], 1.0 / h_sig0, 1.0))
    if clock_sig > 0:
        priors.append((ix["dt"], math.sqrt(k_prior) / clock_sig, n_eff0 / n_pts))
    for a in aircraft:
        priors.append((ix[f"lag_{a}"], math.sqrt(k_prior) / lag_sig, n_eff0 / n_pts))
    if fit_w:
        for nm in ("we", "wn"):
            priors.append((ix[nm], math.sqrt(k_prior) / s.wind_sigma_mps, n_eff0 / n_pts))
    if "c" in ix and s.c_sigma_mps > 0:
        priors.append((ix["c"], math.sqrt(k_prior) / s.c_sigma_mps, n_eff0 / n_pts))
    prior_mean = ({"we": s.atmos.wind_e, "wn": s.atmos.wind_n} if fit_w else {}) | {"c": s.c}
    p_mu = np.array([prior_mean.get(names[i], 0.0) for i, _, _ in priors], dtype=float)
    p_idx = np.array([p[0] for p in priors], dtype=np.intp)
    p_inv = np.array([p[1] for p in priors], dtype=float)
    p_rw = np.array([p[2] for p in priors], dtype=float)

    def resid(x: FloatArray) -> FloatArray:
        r = data_resid(x)
        if not priors:
            return r
        return np.concatenate([r, _quadratic_under_soft_l1((x[p_idx] - p_mu) * p_inv, s.robust_f_scale)])

    lo = np.full(len(names), -np.inf)
    hi = np.full(len(names), np.inf)
    lo[0], hi[0], lo[1], hi[1] = -100_000.0, 100_000.0, -100_000.0, 100_000.0
    if hold_position:  # profiles: the observer stays at (e0, n0) within 1 cm, all else is refitted
        lo[0], hi[0], lo[1], hi[1] = e0 - 0.01, e0 + 0.01, n0 - 0.01, n0 + 0.01
    for f in fams:  # rest frequency within the fit's physical range (10 Hz .. 5 kHz)
        lo[ix[f"lf0_{f.id}"]], hi[ix[f"lf0_{f.id}"]] = math.log(5.0), math.log(10_000.0)
        if drift[f.id]:
            lo[ix[f"rate_{f.id}"]], hi[ix[f"rate_{f.id}"]] = -0.01, 0.01
    if "dt" in ix:
        lo[ix["dt"]], hi[ix["dt"]] = -B, B
    if "dh" in ix:
        lo[ix["dh"]], hi[ix["dh"]] = -max(5.0 * h_sig0, 50.0), max(5.0 * h_sig0, 50.0)
    if "c" in ix:
        lo[ix["c"]], hi[ix["c"]] = 300.0, 360.0
    for a in aircraft:
        lo[ix[f"lag_{a}"]], hi[ix[f"lag_{a}"]] = -6.0 * lag_sig, 6.0 * lag_sig
    if fit_w:
        for nm in ("we", "wn"):
            lo[ix[nm]], hi[ix[nm]] = -40.0, 40.0
    span = hi - lo
    inset = np.where(np.isfinite(span), 1e-9 * np.where(np.isfinite(span), span, 0.0), 0.0)
    x0 = np.clip(x0, lo + inset, hi - inset)
    # Absolute finite-difference steps, well above the retarded-time solve's 1e-8 s tolerance.
    steps = np.array([0.05 if nm in ("e", "n", "dh", "we", "wn") else 1e-4 if nm == "dt" or nm.startswith("lag_")
                      else 1e-3 if nm == "c" else 1e-7 for nm in names])
    res = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", loss="soft_l1",
                        f_scale=s.robust_f_scale, diff_step=steps / np.maximum(np.abs(x0), 1.0),
                        max_nfev=200 * len(names))
    n_data = sum(m.size for m in masks.values())
    prior_r = (res.x[p_idx] - p_mu) * p_inv          # untransformed prior residuals
    clipped = np.concatenate([np.minimum(res.fun[:n_data] ** 2, s.outlier_clip**2), prior_r**2])
    chi2 = float(np.sum(clipped))
    dof = n_pts - len(names) + len(priors)
    # Serially correlated residuals (10 points/s that wander together) carry far less
    # information than their count: weight each family's rows by 1/corr_factor of its
    # single-pass fit, i.e. use its effective number of independent points, for the
    # covariance and its chi2 scaling. Without this the ellipse on real passes came
    # out ~120 m x 10 m while the error was ~500 m. Prior rows: their exact (linear)
    # Jacobian, weighted so each prior adds ~1/sigma^2 to the information.
    J = res.jac.copy()
    if priors:
        J[n_data:] = 0.0
        J[n_data + np.arange(len(priors)), p_idx] = p_inv
    rw = np.concatenate([np.full(masks[f.id].size, 1.0 / max(f.fit.corr_factor, 1.0)) for f in fams]
                        + [p_rw])
    n_eff = n_eff0
    dof_eff = n_eff - len(names) + len(priors)
    chi2_eff = float(np.sum(clipped * rw))
    scale = max(chi2_eff / dof_eff, 1.0) if dof_eff > 0 else max(chi2 / max(dof, 1), 1.0)
    try:
        cov = np.linalg.inv(J.T @ (J * rw[:, None])) * scale
    except np.linalg.LinAlgError:
        cov = np.full((len(names), len(names)), np.inf)
    try:
        plain = np.linalg.inv(J.T @ J) * max(chi2 / max(dof, 1), 1.0)
        infl = float(np.trace(cov[:2, :2]) / np.trace(plain[:2, :2]))
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
        infl = float("nan")
    e, n, dt, h, c = unpack(res.x)
    hp, hs = (float(v) for v in observer_height(frame, e, n, s))
    if fit_h:
        hs = float(np.sqrt(cov[ix["dh"], ix["dh"]])) if np.isfinite(cov[ix["dh"], ix["dh"]]) else hs
    lat, lon = frame.latlon(e, n)
    fam_chi2, a = {}, 0
    for f in fams:
        m = masks[f.id]
        fam_chi2[f.id] = (float(np.sum(clipped[a:a + m.size])), int(m.sum()))
        a += m.size
    return RefinedFit(float(e), float(n), float(h), float(dt), float(c), float(lat), float(lon), chi2, dof,
                      cov[:2, :2], dict(assignment), {f.id: math.exp(res.x[ix[f"lf0_{f.id}"]]) for f in fams},
                      bool(res.success), str(res.message), seed, fam_chi2, hp, hs, fit_h, infl,
                      chi2_eff=chi2_eff, chi2_scale=scale, lags={a: float(res.x[ix[f"lag_{a}"]]) for a in aircraft},
                      atmos=air(res.x))


def _quadratic_under_soft_l1(z: FloatArray, f_scale: float) -> FloatArray:
    """Residuals whose soft_l1 cost equals z**2 / 2 (the plain least-squares cost), so a
    Gaussian prior keeps its strength in a robust fit: solves
    F^2 (sqrt(1 + (r/F)^2) - 1) = z^2 / 2 for r, keeping the sign of z."""
    u = 1.0 + 0.5 * (z / f_scale) ** 2
    return np.sign(z) * f_scale * np.sqrt(np.maximum(u * u - 1.0, 0.0))


@dataclass
class EllipseMeasure:
    cov_en: FloatArray                 # 2x2, scaled so ellipse_95() draws the measured contour
    radii: list[tuple[float, float]]   # (bearing deg, distance m to the 95 % level; inf = unbounded)
    refits: int
    moved: bool                        # a probe fitted better: the solution was re-solved from it


def measure_ellipse(best: RefinedFit, families: Sequence[FamilyObs], frame: Frame, s: SolverSettings,
                    refit: Any) -> tuple[RefinedFit, EllipseMeasure | None]:
    """The 95 % region measured on the fit itself.

    In each of ``s.ellipse_directions`` directions the observer is held at increasing
    distances (everything else refitted, ``refine(hold_position=True)``) until the
    fit's correlation-weighted chi2 has risen by CHI2_2DOF_95 in the covariance's units
    ((chi2_eff - chi2_eff_best) / chi2_scale_best); the crossing is interpolated as
    a power law between the bracketing probes (the valley is not parabolic: it
    flattens with distance). An ellipse x^T A x = 1 is then fitted through the
    crossing points. A probe that fits better than the solution by > 1 means the
    minimum was not found: ``refit(e, n, dt)`` (unheld) re-solves from it, once.
    Returns (best, measure); measure is None when the probes failed."""
    used = [f for f in families if f.id in best.assignment]
    moved = False
    for _attempt in range(2):
        ref, scale = best.chi2_eff, best.chi2_scale
        if not (np.isfinite(ref) and np.isfinite(scale) and scale > 0):
            return best, None
        nref = 0
        better: tuple[float, float, float, float] | None = None  # (rise, e, n, dt)

        def rise_at(u: FloatArray, r: float) -> float:
            nonlocal nref, better
            e, n = best.e + r * u[0], best.n + r * u[1]
            try:
                fit = refine(used, best.assignment, frame, e, n, best.dt, s, "ellipse", hold_position=True)
            except (SolverError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
                return float("nan")
            finally:
                nref += 1
            d = (fit.chi2_eff - ref) / scale
            if d < -1.0 and (better is None or d < better[0]):
                better = (d, e, n, fit.dt)
            return d

        radii, pts = [], []
        # Probe directions follow the Jacobian ellipse (whitened: w = L v, |v| = 1, so t is
        # in Jacobian sigmas and t = sqrt(5.99) is its 95 % contour): a thin valley (two
        # passes, one degenerate direction) is missed by fixed compass directions. Without a
        # usable Jacobian covariance: compass directions, t in units of ellipse_probe_m.
        try:
            L = np.linalg.cholesky(best.cov_en) if np.isfinite(best.cov_en).all() else None
        except np.linalg.LinAlgError:
            L = None
        for k in range(s.ellipse_directions):
            phi = 2.0 * math.pi * k / s.ellipse_directions
            v = np.array([math.sin(phi), math.cos(phi)])
            if L is not None:
                w, t = L @ v, math.sqrt(CHI2_2DOF_95)
            else:
                w, t = s.ellipse_probe_m * v, 1.0
            wn = float(np.hypot(*w))
            if not wn > 0:
                continue
            u = w / wn
            t_max = s.ellipse_max_m / wn
            t = min(t, t_max)
            d = rise_at(u, t * wn)
            lo = (0.0, 0.0)  # (t, rise) inside the level
            t95 = float("inf")
            for _ in range(7):
                if not np.isfinite(d):
                    break
                if d >= CHI2_2DOF_95:
                    t0, d0 = lo
                    if t0 > 0 and d0 > 0:  # power law d = k t^p through both probes
                        p = math.log(d / d0) / math.log(t / t0)
                        t95 = t0 * (CHI2_2DOF_95 / d0) ** (1.0 / p) if p > 0 else t
                    else:  # parabola from the minimum
                        t95 = t * math.sqrt(CHI2_2DOF_95 / d)
                    t95 = float(np.clip(t95, t0, t))
                    break
                lo = (t, max(d, 0.0))
                if t >= t_max:
                    break
                nxt = t * math.sqrt(CHI2_2DOF_95 / d) * 1.15 if d > 0.3 else 3.0 * t
                t = min(max(nxt, 1.3 * t), 4.0 * t, t_max)
                d = rise_at(u, t * wn)
            r95 = t95 * wn
            ang = math.atan2(u[0], u[1])
            radii.append((math.degrees(ang) % 360.0, r95))
            if np.isfinite(r95):
                pts.append(r95 * u)
        if better is not None and not moved:
            new = refit(better[1], better[2], better[3])
            if new is not None and new.chi2_eff < best.chi2_eff - scale:
                best, moved = new, True
                continue
        break
    unbounded = any(not np.isfinite(r) for _, r in radii)
    if unbounded or len(pts) < 5:
        cov = np.full((2, 2), np.inf)
    else:
        P = np.asarray(pts)
        M = np.c_[P[:, 0] ** 2, 2 * P[:, 0] * P[:, 1], P[:, 1] ** 2]
        a11, a12, a22 = np.linalg.lstsq(M, np.ones(len(P)), rcond=None)[0]
        A = np.array([[a11, a12], [a12, a22]])
        lam = np.linalg.eigvalsh(A)
        if lam.min() <= 0:  # the points are not an ellipse: bound it by the farthest crossing
            rmax = float(np.max(np.hypot(P[:, 0], P[:, 1])))
            A = np.eye(2) / rmax**2
        cov = np.linalg.inv(A) / CHI2_2DOF_95
    return best, EllipseMeasure(cov, radii, nref, moved)


def _mirror_seeds(best: RefinedFit, s: SolverSettings) -> list[tuple[float, float]]:
    """Reflect the solution across each assigned track (ground projection at CPA)."""
    out = []
    o = np.array([best.e, best.n])
    seen: set[str] = set()
    for cand in best.assignment.values():
        if cand.icao24 in seen:
            continue
        seen.add(cand.icao24)
        st, P, V = cand.traj.samples(cand.traj.t0, cand.traj.t1, 1.0)
        if st.size < 2:
            continue
        j = int(np.argmin(np.hypot(P[:, 0] - o[0], P[:, 1] - o[1])))
        u = V[j, :2] / max(np.hypot(*V[j, :2]), 1e-9)
        nrm = np.array([-u[1], u[0]])
        m = o - 2.0 * float((o - P[j, :2]) @ nrm) * nrm
        out.append((float(m[0]), float(m[1])))
    return out


# ---------------------------------------------------------------------------
# Step 4: ambiguity, grading, top level
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    lat: float
    lon: float
    h: float
    dt_clock: float
    c: float
    cov_en: FloatArray
    ellipse_95: tuple[float, float, float]
    chi2: float
    chi2_red: float
    grade: str
    resolved: bool
    mirror: dict | None             # {lat, lon, chi2, delta_chi2} of the competing solution
    families_used: list[int]
    aircraft_used: list[str]
    assignment: dict[int, str]      # family id -> icao24
    f0: dict[int, float]
    unexplained: list[int]          # family ids without any gated candidate
    candidates: list[Candidate] = field(repr=False)
    grid: GridResult | None = field(repr=False, default=None)
    fits: list[RefinedFit] = field(repr=False, default_factory=list)
    frame: Frame | None = field(repr=False, default=None)
    best: RefinedFit | None = field(repr=False, default=None)
    h_source: str = "constant"
    h_prior: float = float("nan")    # elevation model at the solution
    h_sigma: float = float("nan")    # 1-sigma of h (posterior if fitted)
    h_fitted: bool = False
    chi2_excess: float = float("nan")  # joint chi2_red / the families' single-pass chi2_red
    ellipse_jacobian: tuple[float, float, float] | None = None  # the Jacobian estimate, for comparison
    ellipse_measure: EllipseMeasure | None = field(repr=False, default=None)

    def predict(self, family_id: int, t_media: ArrayLike, k: ArrayLike = 1.0) -> FloatArray:
        """Predicted frequency of a family's assigned aircraft at the solution (spec UX rule)."""
        assert self.best is not None and self.frame is not None
        cand = self.best.assignment[family_id]
        obs = self.frame.observer(self.best.e, self.best.n, self.h)
        shift = self.best.time_shift(cand.icao24)
        logD, valid, _ = log_doppler(cand.traj, obs, np.asarray(t_media, dtype=float) + shift, self.c,
                                     atmos=self.best.atmos)
        return np.where(valid, np.asarray(k, dtype=float) * self.f0[family_id] * np.exp(logD), np.nan)

    def as_dict(self) -> dict:
        a, b, ang = self.ellipse_95
        return {
            "lat": self.lat, "lon": self.lon, "h": self.h, "h_source": self.h_source, "h_prior": self.h_prior,
            "h_sigma": self.h_sigma, "h_fitted": self.h_fitted, "dt_clock": self.dt_clock, "c": self.c,
            "ellipse_method": "measured" if self.ellipse_measure is not None else "jacobian",
            "ellipse_95_jacobian": (None if self.ellipse_jacobian is None else
                                    dict(zip(("semi_major_m", "semi_minor_m", "bearing_deg"), self.ellipse_jacobian))),
            "ellipse_radii_95": (None if self.ellipse_measure is None else
                                 [{"bearing_deg": b_, "distance_m": r_} for b_, r_ in self.ellipse_measure.radii]),
            "cov_en": self.cov_en.tolist(), "ellipse_95": {"semi_major_m": a, "semi_minor_m": b,
                                                          "bearing_deg": ang},
            "adsb_lag_s": dict(self.best.lags) if self.best is not None else {},
            "atmosphere": (None if self.best is None or self.best.atmos is None else
                           {"temp_c": self.best.atmos.temp_c, "wind_e": self.best.atmos.wind_e,
                            "wind_n": self.best.atmos.wind_n, "lapse": self.best.atmos.lapse,
                            "source": self.best.atmos.source}),
            "chi2": self.chi2, "chi2_red": self.chi2_red, "chi2_excess": self.chi2_excess, "grade": self.grade, "resolved": self.resolved,
            "mirror": self.mirror, "families_used": self.families_used, "aircraft_used": self.aircraft_used,
            "assignment": {str(k): v for k, v in self.assignment.items()},
            "f0": {str(k): v for k, v in self.f0.items()}, "unexplained": self.unexplained,
        }

    def to_geojson(self, n_vertices: int = 72) -> dict:
        """Solution point, 95% ellipse polygon and mirror point (WGS84 lon/lat)."""
        assert self.best is not None and self.frame is not None
        a, b, ang = self.ellipse_95
        feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
                  "properties": {"kind": "solution", **{k: v for k, v in self.as_dict().items()
                                                         if k in ("grade", "dt_clock", "chi2_red", "resolved")}}}]
        if np.isfinite(a):
            th = np.linspace(0, 2 * np.pi, n_vertices + 1)
            br = math.radians(ang)
            maj = np.array([math.sin(br), math.cos(br)])
            mnr = np.array([math.cos(br), -math.sin(br)])
            pts = self.best.e + 0 * th, self.best.n + 0 * th
            e = pts[0] + a * np.cos(th) * maj[0] + b * np.sin(th) * mnr[0]
            n = pts[1] + a * np.cos(th) * maj[1] + b * np.sin(th) * mnr[1]
            lat, lon = self.frame.latlon(e, n)
            feats.append({"type": "Feature", "properties": {"kind": "ellipse_95", "semi_major_m": a,
                                                            "semi_minor_m": b, "bearing_deg": ang},
                          "geometry": {"type": "Polygon", "coordinates": [np.c_[lon, lat].tolist()]}})
        if self.mirror:
            feats.append({"type": "Feature", "properties": {"kind": "mirror", **self.mirror},
                          "geometry": {"type": "Point", "coordinates": [self.mirror["lon"], self.mirror["lat"]]}})
        return {"type": "FeatureCollection", "features": feats}


def grade_solution(n_aircraft: int, resolved: bool, semi_major: float, chi2_excess: float = 1.0,
                   max_chi2_excess: float = 25.0, contradicted: int = 0) -> str:
    """``chi2_excess``: the joint fit's chi2_red over the families' own single-pass
    chi2_red (1 = the position explains every pass as well as its own fit does).
    ``contradicted``: trustworthy (strictly gated) families that had candidate
    aircraft but are left out of the solution, i.e. the position contradicts them."""
    if not resolved or not semi_major < 2000.0 or not chi2_excess <= max_chi2_excess or contradicted:
        return "C"
    if n_aircraft >= 2 and semi_major < 500.0:
        return "A"
    return "B"


def solve(families: Sequence[FamilyObs], tracks: Sequence[Track], epoch: float, region: Region,
          settings: SolverSettings | None = None, progress=None) -> Solution:
    """Locate the observer from fitted tone families and ADS-B tracks.

    ``epoch`` is the file's UTC (Unix seconds) at media time 0. ``progress(stage,
    done, total)`` (total 0: no count) reports the stages for a UI.
    """
    s = settings or SolverSettings()
    if s.atmos is not None:  # rough uses of c (gating, grid spans) take the surface value
        s = replace(s, c=s.atmos.c_surface)
    if s.clock_sigma_s > 0 and s.clock_bound_s > 0:  # the prior makes offsets beyond 4 sigma implausible
        s = replace(s, clock_bound_s=min(s.clock_bound_s, 4.0 * s.clock_sigma_s))
    report = progress or (lambda stage, done, total: None)
    lat_min, lon_min, lat_max, lon_max = region.as_tuple()
    frame = Frame((lat_min + lat_max) / 2, (lon_min + lon_max) / 2, epoch)
    dropped: list[int] = []
    good = [f for f in families if f.fit.grade in ("A", "B") and not f.loose]
    if s.reliable_only_min and len(good) >= s.reliable_only_min:
        dropped = [f.id for f in families if f not in good]
        families = good
    if len(families) > s.max_families:
        rank = sorted(families, key=lambda f: ("ABC".index(f.fit.grade), f.loose, -f.t.size))
        keep = {f.id for f in rank[: s.max_families]}
        dropped += [f.id for f in families if f.id not in keep]
        families = [f for f in families if f.id in keep]
    report("selecting candidate aircraft", 0, 0)
    gated = gate_candidates(families, tracks, frame, region, s)
    excluded = [replace(c, included=False) for c in gated if c.icao24 in s.exclude_aircraft]
    cands = [c for c in gated if c.icao24 not in s.exclude_aircraft]
    with_c = {c.family_id for c in cands}
    used = [f for f in families if f.id in with_c]
    unexplained = [f.id for f in families if f.id not in with_c] + dropped
    if not used:
        raise SolverError("no ADS-B aircraft passes the speed/time/area gates for any tone family")

    grid = grid_search(used, cands, frame, region, s, progress=report)
    seeds = _grid_seeds(grid, s.n_seeds, s.seed_separation_m)
    if not seeds:
        raise SolverError("no grid cell is explained by the candidate aircraft")
    # Each trustworthy family also seeds at its own best cell: the total-chi2 minima
    # can be dominated by loose (junk) families, and a hypothesis that never
    # contains the good family cannot win however the hypotheses are ranked.
    E, N = (a.ravel() for a in np.meshgrid(grid.e, grid.n))
    for fi, f in enumerate(used):
        if f.loose:
            continue
        fc = np.where(grid.assign[fi] >= 0, grid.family_chi2[fi], np.inf).ravel()
        if not np.isfinite(fc).any():
            continue
        i = int(np.argmin(fc))
        if all(math.hypot(E[i] - e, N[i] - n) >= s.seed_separation_m for e, n, _ in seeds):
            seeds.append((float(E[i]), float(N[i]), i))

    def assignment_at(flat: int) -> tuple[dict[int, Candidate], float]:
        asg, dts = {}, []
        for fi, f in enumerate(used):
            ci = int(grid.assign[fi].ravel()[flat])
            if ci >= 0:
                asg[f.id] = cands[ci]
                if not f.loose:
                    dts.append(float(grid.dt[fi].ravel()[flat]))
        return asg, float(np.mean(dts)) if dts else 0.0

    re0, re1, rn0, rn1 = _region_en(frame, region)

    def try_refine(asg: dict[int, Candidate], e: float, n: float, dt: float, seed: str,
                   ss: SolverSettings | None = None) -> RefinedFit | None:
        if not asg:
            return None
        try:
            r = refine(used, asg, frame, e, n, dt, ss or s, seed)
        except (SolverError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return None
        if not np.isfinite(r.chi2):
            return None
        # The search region is a constraint: a refinement that wandered out of it
        # (seen on a real recording: 103 km away from a 25 km region) is rejected.
        m = s.region_margin_m
        if not (re0 - m <= r.e <= re1 + m and rn0 - m <= r.n <= rn1 + m):
            return None
        return r

    def prune(fit: RefinedFit) -> RefinedFit:
        """Drop families the hypothesis cannot explain.

        1. An outlier: chi2_red far above the median of the other families.
        2. Leave-one-out: while the fit is poor, drop the family whose removal
           improves chi2_red the most, if that at least halves it (catches
           several bad families dragging the solution together).
        """
        while len(fit.family_chi2) >= 2:
            red = {fid: c / max(k, 1) for fid, (c, k) in fit.family_chi2.items()}
            med = float(np.median(list(red.values())))
            worst = max(red, key=red.__getitem__)
            new = None
            if red[worst] > max(s.prune_factor * med, s.prune_floor):
                new = try_refine({k: v for k, v in fit.assignment.items() if k != worst}, fit.e, fit.n,
                                 fit.dt, fit.seed)
            elif fit.chi2_red > s.prune_floor:
                trials = [try_refine({k: v for k, v in fit.assignment.items() if k != fid}, fit.e, fit.n,
                                     fit.dt, fit.seed) for fid in fit.assignment]
                trials = [t for t in trials if t is not None]
                cand = min(trials, key=lambda t: t.chi2_red, default=None)
                if cand is not None and cand.chi2_red < 0.5 * fit.chi2_red:
                    new = cand
            if new is None:
                break
            fit = new
        return fit

    hyps = []
    for i, (e, n, flat) in enumerate(seeds):
        report(f"refining position {i + 1}/{len(seeds)}", i, len(seeds))
        asg, dt0 = assignment_at(flat)
        r = try_refine(asg, e, n, dt0, "grid")
        if r is not None:
            hyps.append(prune(r))
    if not hyps:
        raise SolverError("refinement failed at every grid minimum")
    # Admissible hypotheses fit about as well as the best one does; among those,
    # explaining more trustworthy (strictly gated) families wins, then more
    # families in total, then lower chi2. Loose families (failed or vague fits)
    # must not outvote a good one: on a real recording two junk families on one
    # aircraft beat the only grade-B pass when every family counted the same.
    strict = {f.id for f in used if not f.loose}
    floor = min(h.chi2_red for h in hyps)
    limit = max(s.max_chi2_red, 3.0 * floor)
    hyps.sort(key=lambda r: (r.chi2_red > limit, -len(strict & set(r.assignment)), -len(r.assignment), r.chi2))
    best = hyps[0]

    # Re-admit families the grid left out (e.g. loose fits scored with dt = 0 while
    # the clock is off): try each of their candidates jointly with the best hypothesis.
    report("re-checking left-out families", 0, 0)
    for f in used:
        if f.id in best.assignment:
            continue
        accepted = []
        mine = sorted((c for c in cands if c.family_id == f.id), key=lambda c: abs(c.speed_match - 1.0))
        for cand in mine[: s.max_readmit_candidates]:
            r = try_refine({**best.assignment, f.id: cand}, best.e, best.n, best.dt, best.seed)
            if r is None or f.id not in r.assignment:  # refine drops a family out of its aircraft's range
                continue
            red = {fid: c / max(k, 1) for fid, (c, k) in r.family_chi2.items()}
            others = [v for fid, v in red.items() if fid != f.id]
            if (red[f.id] <= max(s.prune_factor * float(np.median(others)), s.prune_floor)
                    and r.chi2_red <= max(2.0 * best.chi2_red, s.prune_floor)):
                accepted.append(r)
        if accepted:
            best = min(accepted, key=lambda r: r.chi2)

    sh = s
    report("checking the mirror position", 0, 0)
    # Mirror / alternative positions under the same hypothesis (same families, same aircraft).
    alts = [try_refine(best.assignment, e, n, best.dt, "grid", sh) for e, n, _ in seeds
            if math.hypot(e - best.e, n - best.n) > s.seed_separation_m / 2]
    alts += [try_refine(best.assignment, e, n, best.dt, "mirror", sh) for e, n in _mirror_seeds(best, s)]
    alts = [r for r in alts if r is not None]
    if any(r.chi2 < best.chi2 for r in alts):  # a seed under this hypothesis found a better spot
        alts.append(best)
        alts.sort(key=lambda r: r.chi2)
        best = alts.pop(0)
    jac_ellipse = best.ellipse_95()
    measured = None
    if s.measure_ellipse:
        report("measuring the 95 % ellipse", 0, 0)
        best, measured = measure_ellipse(best, used, frame, s,
                                         lambda e, n, dt: try_refine(best.assignment, e, n, dt, "ellipse"))
        if measured is not None:
            best = replace(best, cov_en=measured.cov_en)
    fits = [best] + sorted(alts, key=lambda r: r.chi2) + [h for h in hyps[1:]]
    a, b, ang = best.ellipse_95()
    sep = max(250.0, 2.0 * a if np.isfinite(a) else 250.0)
    rivals = sorted((r for r in alts if math.hypot(r.e - best.e, r.n - best.n) > sep), key=lambda r: r.chi2)
    mirror = None
    resolved = True
    if rivals:
        r = rivals[0]
        dchi2 = (r.chi2 - best.chi2) / max(best.chi2_red, 1.0)
        mirror = {"lat": r.lat, "lon": r.lon, "chi2": r.chi2, "delta_chi2": dchi2, "seed": r.seed}
        resolved = dchi2 > s.mirror_dchi2
    unexplained = sorted(unexplained + [f.id for f in used if f.id not in best.assignment])

    # Per-candidate chi-square at the solution (Flights table column).
    obs = frame.observer(best.e, best.n, best.h)
    fam_by_id = {f.id: f for f in used}
    for cand in cands:
        f = fam_by_id[cand.family_id]
        logD, valid, _ = log_doppler(cand.traj, obs, f.t + best.time_shift(cand.icao24), best.c,
                                     atmos=best.atmos)
        w = (f.f / f.sigma) ** 2 * valid
        if valid.mean() < s.min_valid_frac:
            cand.chi2 = float("inf")
            continue
        y = np.log(f.f / f.k) - logD
        lf0 = float(np.median(y[valid]))  # robust: glitches must not drag f0
        cand.chi2 = float(np.sum(np.where(valid, np.minimum(w * (y - lf0) ** 2, s.outlier_clip**2), 0.0)))

    aircraft = sorted({c.icao24 for c in best.assignment.values()})
    own = [fam_by_id[fid].fit for fid in best.assignment]
    own_red = sum(f.chi2 for f in own) / max(sum(f.dof for f in own), 1)
    chi2_excess = best.chi2_red / max(own_red, 1.0)
    return Solution(
        lat=best.lat, lon=best.lon, h=best.h, dt_clock=best.dt, c=best.c, cov_en=best.cov_en,
        ellipse_95=(a, b, ang), chi2=best.chi2, chi2_red=best.chi2_red,
        grade=grade_solution(len(aircraft), resolved, a, chi2_excess, s.max_chi2_excess,
                             contradicted=sum(1 for f in used if not f.loose and f.id not in best.assignment)),
        resolved=resolved, mirror=mirror,
        families_used=sorted(best.assignment), aircraft_used=aircraft,
        assignment={fid: c.icao24 for fid, c in best.assignment.items()}, f0=best.f0,
        unexplained=unexplained, candidates=list(cands) + excluded, grid=grid, fits=fits, frame=frame, best=best,
        h_source=getattr(elevation_model(s), "source", "?"), h_prior=best.h_prior, h_sigma=best.h_sigma,
        h_fitted=best.h_fitted, chi2_excess=chi2_excess, ellipse_jacobian=jac_ellipse, ellipse_measure=measured)
