"""Moving-source Doppler model for a stationary observer in still air.

Conventions
-----------
* ``tau`` is emission time, ``t`` is reception time (seconds, common epoch).
* ``r(tau) = aircraft(tau) - observer``; ``r_hat`` points observer -> aircraft.
* Radial speed ``v_r = v . r_hat`` is negative while the source approaches.
* Received frequency::

      f(t) = f0 / (1 + v_r(tau) / c),    t = tau + |r(tau)| / c

  so the pitch is high on approach and low on departure.

Straight-line, constant-velocity source (speed ``v``, CPA slant distance ``d``,
emission CPA time ``tau0``), with ``s = tau - tau0``::

      v_r(s) = v^2 s / sqrt(d^2 + v^2 s^2)

The retarded-time equation ``t = tau0 + s + sqrt(d^2 + v^2 s^2) / c`` is a
quadratic in ``s``; the root with ``s <= t - tau0`` (sound emitted before it is
heard) is::

      s = (T - sqrt(M^2 T^2 + (d/c)^2 (1 - M^2))) / (1 - M^2),  T = t - tau0, M = v/c
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .atmosphere import SPEED_OF_SOUND_15C

FloatArray = NDArray[np.float64]
TrajectoryFn = Callable[[FloatArray], FloatArray]  # (N,) emission times -> (N, 3) ENU metres


# ---------------------------------------------------------------------------
# Closed-form straight-line model
# ---------------------------------------------------------------------------


def retarded_offset(
    t: ArrayLike, tau0: float, v: float, d: float, c: float = SPEED_OF_SOUND_15C
) -> FloatArray:
    """Emission offset ``s = tau - tau0`` for sound received at time ``t``."""
    T = np.asarray(t, dtype=float) - tau0
    m = v / c
    one_m2 = 1.0 - m * m
    return (T - np.sqrt((m * T) ** 2 + (d / c) ** 2 * one_m2)) / one_m2


def radial_speed(s: ArrayLike, v: float, d: float) -> FloatArray:
    """Line-of-sight speed ``v . r_hat`` (m/s) at emission offset ``s``."""
    s = np.asarray(s, dtype=float)
    return v * v * s / np.sqrt(d * d + (v * s) ** 2)


def cpa_reception_time(tau0: float, d: float, c: float = SPEED_OF_SOUND_15C) -> float:
    """Time at which the closest-approach sound reaches the observer."""
    return tau0 + d / c


def doppler_frequency(
    t: ArrayLike,
    f0: float,
    v: float,
    d: float,
    tau0: float,
    c: float = SPEED_OF_SOUND_15C,
    k: ArrayLike = 1.0,
    f0_rate: float = 0.0,
) -> FloatArray:
    """Received frequency of harmonic ``k`` at reception times ``t``.

    ``f0_rate`` (Hz/s) lets the rest frequency drift linearly in *emission* time,
    ``f0 + f0_rate * (tau - tau0)``, to absorb a slow engine power change.
    """
    s = retarded_offset(t, tau0, v, d, c)
    return np.asarray(k, dtype=float) * (f0 + f0_rate * s) / (1.0 + radial_speed(s, v, d) / c)


# ---------------------------------------------------------------------------
# General trajectory model (used for ADS-B tracks and to verify the closed form)
# ---------------------------------------------------------------------------


def retarded_time(
    t: ArrayLike,
    position: TrajectoryFn,
    observer: ArrayLike,
    c: float = SPEED_OF_SOUND_15C,
    velocity: TrajectoryFn | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> FloatArray:
    """Solve ``t = tau + |position(tau) - observer| / c`` for emission time ``tau``.

    Newton's method on g(tau) = tau + |r(tau)|/c - t, with g' = 1 + v_r/c > 0 for
    subsonic sources, so the root is unique. ``velocity`` is optional; without it
    the radial speed is taken by central differences.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    obs = np.asarray(observer, dtype=float)
    # Absolute tolerance cannot beat float64 resolution of large epochs (UTC s).
    tol = max(tol, 8 * np.finfo(float).eps * float(np.max(np.abs(t), initial=1.0)))

    tau = t - np.linalg.norm(position(t) - obs, axis=-1) / c
    for _ in range(max_iter):
        r = position(tau) - obs
        dist = np.linalg.norm(r, axis=-1)
        g = tau + dist / c - t
        if velocity is not None:
            v_r = np.einsum("ij,ij->i", velocity(tau), r) / dist
        else:
            h = 1e-3
            v_r = (
                np.linalg.norm(position(tau + h) - obs, axis=-1)
                - np.linalg.norm(position(tau - h) - obs, axis=-1)
            ) / (2 * h)
        step = g / (1.0 + v_r / c)
        tau = tau - step
        # Times whose trajectory is undefined (NaN, e.g. emission before an ADS-B
        # track starts) stay NaN instead of blocking convergence of the rest.
        done = ~np.isfinite(step) | (np.abs(step) < tol)
        if done.all():
            if not np.isfinite(tau).any():
                raise RuntimeError("retarded_time: trajectory undefined at every requested time")
            return tau
    raise RuntimeError("retarded_time did not converge (supersonic source or bad trajectory?)")


def frequency_from_trajectory(
    t: ArrayLike,
    f0: float,
    position: TrajectoryFn,
    velocity: TrajectoryFn,
    observer: ArrayLike,
    c: float = SPEED_OF_SOUND_15C,
    k: ArrayLike = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Received frequency for an arbitrary trajectory. Returns ``(f, tau)``."""
    obs = np.asarray(observer, dtype=float)
    tau = retarded_time(t, position, obs, c, velocity)
    r = position(tau) - obs
    v_r = np.einsum("ij,ij->i", velocity(tau), r) / np.linalg.norm(r, axis=-1)
    return np.asarray(k, dtype=float) * f0 / (1.0 + v_r / c), tau


# ---------------------------------------------------------------------------
# Single-pass fit
# ---------------------------------------------------------------------------

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "f0": (10.0, 5000.0),
    "v": (30.0, 320.0),
    "d": (20.0, 20_000.0),
    "f0_rate": (-np.inf, np.inf),
}

# Grade thresholds: relative 1-sigma errors on v and d.
GRADE_A = {"v": 0.03, "d": 0.10, "chi2_red": 3.0, "outliers": 0.05}
GRADE_B = {"v": 0.10, "d": 0.30, "chi2_red": 10.0, "outliers": 0.20}


@dataclass
class DopplerFit:
    """Result of fitting one harmonic family to the straight-line model."""

    f0: float
    v: float
    d: float
    tau0: float
    f0_rate: float
    c: float
    param_names: tuple[str, ...]
    cov: FloatArray
    chi2: float
    dof: int
    r2: float
    t_range: tuple[float, float]
    success: bool
    message: str
    residuals: FloatArray = field(repr=False)
    sigma: FloatArray = field(repr=False)
    outlier_fraction: float = 0.0  # points beyond 5 robust sigma (excluded from chi2)
    corr_factor: float = 1.0  # covariance inflation for serially correlated residuals
    at_bound: bool = False  # v or d pinned to a fit bound: the minimum is not a real one

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    @property
    def t_cpa(self) -> float:
        """Reception time of the closest-approach sound."""
        return cpa_reception_time(self.tau0, self.d, self.c)

    @property
    def mach(self) -> float:
        return self.v / self.c

    def stderr(self, name: str) -> float:
        """1-sigma standard error of a fitted parameter (0 if it was held fixed)."""
        if name == "t_cpa":
            # t_cpa = tau0 + d/c: propagate through the covariance.
            i, j = self.param_names.index("tau0"), self.param_names.index("d")
            g = np.zeros(len(self.param_names))
            g[i], g[j] = 1.0, 1.0 / self.c
            return float(np.sqrt(g @ self.cov @ g))
        if name not in self.param_names:
            return 0.0
        i = self.param_names.index(name)
        return float(np.sqrt(self.cov[i, i]))

    @property
    def cpa_covered(self) -> bool:
        """True when the CPA lies inside the observed time span."""
        return self.t_range[0] < self.t_cpa < self.t_range[1]

    @property
    def cpa_near(self) -> bool:
        """CPA within the observed span widened by half its length on each side:
        a fit that puts the closest approach far outside the data is an
        extrapolation, not a measurement."""
        t0, t1 = self.t_range
        half = 0.5 * (t1 - t0)
        return t0 - half < self.t_cpa < t1 + half

    @property
    def grade(self) -> str:
        rel_v = self.stderr("v") / self.v
        rel_d = self.stderr("d") / self.d
        if self.at_bound:
            return "C"
        if (
            self.success
            and self.cpa_covered
            and rel_v < GRADE_A["v"]
            and rel_d < GRADE_A["d"]
            and self.chi2_red < GRADE_A["chi2_red"]
            and self.outlier_fraction < GRADE_A["outliers"]
        ):
            return "A"
        if (
            self.success
            and self.cpa_near
            and rel_v < GRADE_B["v"]
            and rel_d < GRADE_B["d"]
            and self.chi2_red < GRADE_B["chi2_red"]
            and self.outlier_fraction < GRADE_B["outliers"]
        ):
            return "B"
        return "C"

    def predict(self, t: ArrayLike, k: ArrayLike = 1.0) -> FloatArray:
        return doppler_frequency(t, self.f0, self.v, self.d, self.tau0, self.c, k, self.f0_rate)

    def params(self) -> FloatArray:
        return np.array([getattr(self, n) for n in self.param_names])

    def as_dict(self) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = {}
        for n in ("f0", "v", "d", "tau0", "f0_rate"):
            out[n] = getattr(self, n)
            out[f"{n}_err"] = self.stderr(n)
        out.update(
            t_cpa=self.t_cpa,
            t_cpa_err=self.stderr("t_cpa"),
            chi2_red=self.chi2_red,
            outlier_fraction=self.outlier_fraction,
            r2=self.r2,
            grade=self.grade,
            cpa_covered=self.cpa_covered,
        )
        return out


def _central_jacobian(fun: Callable[[FloatArray], FloatArray], p: FloatArray) -> FloatArray:
    """Central-difference Jacobian with relative steps."""
    h = 1e-6 * np.maximum(np.abs(p), 1.0)
    cols = []
    for j in range(p.size):
        e = np.zeros_like(p)
        e[j] = h[j]
        cols.append((fun(p + e) - fun(p - e)) / (2.0 * h[j]))
    return np.column_stack(cols)


def _prepare(t: ArrayLike, f: ArrayLike, k: ArrayLike | None, sigma: ArrayLike | None):
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    k = np.ones_like(t) if k is None else np.broadcast_to(np.asarray(k, dtype=float), t.shape)
    s = None if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), t.shape)
    ok = np.isfinite(t) & np.isfinite(f) & (k > 0)
    if s is not None:
        ok &= np.isfinite(s) & (s > 0)
    t, f, k = t[ok], f[ok], k[ok]
    s = None if s is None else s[ok]
    order = np.argsort(t)
    return t[order], f[order], k[order], None if s is None else s[order]


def correlation_factor(r: FloatArray, max_lag: int = 400) -> float:
    """Variance inflation 1 + 2 sum(rho_k) for a time-ordered residual series.

    Real tones wander slowly (engine RPM, turbulence) and overlapping STFT
    frames share samples, so neighbouring residuals are not independent and a
    fit that treats them so overstates its precision by sqrt of this factor.
    The sum runs over the initial positive sequence of the autocorrelation
    (stops at the first non-positive lag), capped at n/10.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    if n < 20:
        return 1.0
    x = r - r.mean()
    var = float(x @ x) / n
    if var <= 0:
        return 1.0
    tot = 0.0
    for k in range(1, min(max_lag, n // 4)):
        rho = float(x[:-k] @ x[k:]) / (n * var)
        if rho <= 0.0:
            break
        tot += rho
    return float(min(1.0 + 2.0 * tot, max(1.0, n / 10.0)))


def initial_guess(
    t: ArrayLike, f: ArrayLike, k: ArrayLike | None = None, c: float = SPEED_OF_SOUND_15C
) -> dict[str, float]:
    """Model-free starting point from asymptotes and the slope at the f0 crossing.

    * f_hi, f_lo: medians of the first/last 10% of points.
    * f0 = 2 f_hi f_lo / (f_hi + f_lo)  (exact for true asymptotes)
    * v  = c (f_hi - f_lo) / (f_hi + f_lo)
    * t_c: reception time where the curve crosses f0; slope there ~ -f0 v^2 / (c d)
    * tau0 = t_c - d / c
    Asymptotes that are not reached bias v low; the least-squares fit fixes that.
    """
    t, f, k, _ = _prepare(t, f, k, None)
    if t.size < 8:
        raise ValueError("need at least 8 points to fit a Doppler curve")
    fn = f / k
    m = max(3, t.size // 10)
    f_hi, f_lo = float(np.median(fn[:m])), float(np.median(fn[-m:]))
    if f_hi <= f_lo:  # no approach-to-departure drop visible
        f0 = float(np.median(fn))
        v = 100.0
    else:
        f0 = 2.0 * f_hi * f_lo / (f_hi + f_lo)
        v = float(np.clip(c * (f_hi - f_lo) / (f_hi + f_lo), *DEFAULT_BOUNDS["v"]))

    above = fn - f0 > 0
    cross = np.nonzero(above[:-1] & ~above[1:])[0]  # high -> low crossings
    if cross.size:
        # Linear interpolation for each crossing; take the median (noise makes several).
        i = cross
        tc = t[i] + (fn[i] - f0) * (t[i + 1] - t[i]) / (fn[i] - fn[i + 1])
        t_c = float(np.median(tc))
    else:
        t_c = float(t[np.argmin(np.abs(fn - f0))])

    def local_slope(half_width: float) -> float | None:
        sel = np.abs(t - t_c) <= half_width
        if np.count_nonzero(sel) < 5:
            return None
        return float(np.polyfit(t[sel] - t_c, fn[sel], 1)[0])

    span = float(t[-1] - t[0])
    d = 2000.0
    slope = local_slope(0.05 * span)
    for _ in range(3):  # shrink the window towards the CPA width ~ d / v
        if slope is None or slope >= 0:
            break
        d = float(np.clip(f0 * v * v / (c * abs(slope)), *DEFAULT_BOUNDS["d"]))
        refined = local_slope(max(0.5 * d / v, 3.0 * float(np.median(np.diff(t)))))
        if refined is None:
            break
        slope = refined
    if slope is not None and slope < 0:
        d = float(np.clip(f0 * v * v / (c * abs(slope)), *DEFAULT_BOUNDS["d"]))

    return {"f0": f0, "v": v, "d": d, "tau0": t_c - d / c, "f0_rate": 0.0}


def fit_doppler(
    t: ArrayLike,
    f: ArrayLike,
    k: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    *,
    c: float = SPEED_OF_SOUND_15C,
    fit_rate: bool = False,
    loss: str = "soft_l1",
    f_scale: float = 3.0,
    x0: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    absolute_sigma: bool = False,
    fixed: dict[str, float] | None = None,
) -> DopplerFit:
    """Fit one harmonic family to the straight-line constant-velocity model.

    Parameters
    ----------
    t, f : reception times (s) and measured frequencies (Hz).
    k : harmonic number per point (default 1). Points from several harmonics of
        the same source may be mixed; they share v, d, tau0.
    sigma : 1-sigma frequency error per point (Hz). If omitted it is estimated
        robustly from a first unweighted pass.
    fit_rate : also fit a linear f0 drift (Hz/s, in emission time).
    loss, f_scale : passed to ``scipy.optimize.least_squares``; residuals are in
        units of sigma, so ``f_scale=3`` starts down-weighting beyond 3 sigma.
    absolute_sigma : if False (default) the covariance is scaled by the reduced
        chi-square of the inliers, as in ``scipy.optimize.curve_fit``.
    fixed : parameters held at a value (the GUI's "lock"), e.g. ``{"v": 82.0}``:
        bounded to +-1e-6 relative and exempt from the at-bound check.
    """
    t, f, k, sig = _prepare(t, f, k, sigma)
    names = ("f0", "v", "d", "tau0") + (("f0_rate",) if fit_rate else ())
    guess = initial_guess(t, f, k, c) if x0 is None else {**initial_guess(t, f, k, c), **x0}

    # Work relative to a local epoch so tau0 is O(100 s), not O(1e9 s).
    t_ref = float(np.median(t))
    tl = t - t_ref
    fixed = {k_: float(v_) for k_, v_ in (fixed or {}).items()}
    b = {**DEFAULT_BOUNDS, **(bounds or {}),
         **{k_: (v_ - 1e-6 * max(abs(v_), 1.0), v_ + 1e-6 * max(abs(v_), 1.0)) for k_, v_ in fixed.items()}}
    if fixed:
        guess = {**guess, **fixed}
    b.setdefault("tau0", (tl[0] - 600.0 + t_ref, tl[-1] + 600.0 + t_ref))
    lo = np.array([b[n][0] - (t_ref if n == "tau0" else 0.0) for n in names])
    hi = np.array([b[n][1] - (t_ref if n == "tau0" else 0.0) for n in names])
    p0 = np.array([guess[n] - (t_ref if n == "tau0" else 0.0) for n in names])
    span = hi - lo
    inset = np.where(np.isfinite(span), 1e-6 * span, 0.0)
    p0 = np.clip(p0, lo + inset, hi - inset)

    def model(p: FloatArray) -> FloatArray:
        f0, v, d, tau0 = p[:4]
        rate = p[4] if fit_rate else 0.0
        return doppler_frequency(tl, f0, v, d, tau0, c, k, rate)

    if sig is None:
        first = least_squares(lambda p: model(p) - f, p0, bounds=(lo, hi), x_scale="jac")
        r0 = model(first.x) - f
        mad = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
        sig = np.full_like(f, max(mad, 1e-9 * float(np.median(f))))
        p0 = first.x

    def resid(p: FloatArray) -> FloatArray:
        return (model(p) - f) / sig

    def solve(start: FloatArray):
        return least_squares(resid, np.clip(start, lo + inset, hi - inset), bounds=(lo, hi),
                             loss=loss, f_scale=f_scale, x_scale="jac", method="trf")

    def lmeds(r) -> float:  # selection criterion between starts: robust, scale-free
        return float(np.median(np.abs(resid(r.x))))

    res = solve(p0)
    if x0 is None:
        # Multi-start: on real (wandering) tones the heuristic guess can fall
        # into a far-away local minimum (v, d at bounds, CPA off the data). Try
        # other CPA distances, and other CPA times: where the rest-frequency-
        # normalised tone crosses the middle of its range, and the span thirds.
        i_d, i_t = names.index("d"), names.index("tau0")
        starts = []
        for d_mult in (0.25, 4.0):
            alt = p0.copy()
            alt[i_d] = p0[i_d] * d_mult
            starts.append(alt)
        fk = f / k
        w = max(5, fk.size // 20)
        sm = np.convolve(fk, np.ones(w) / w, mode="valid")  # t is sorted
        tm = tl[w // 2 : w // 2 + sm.size]
        f_lo, f_hi = np.percentile(sm, 5), np.percentile(sm, 95)
        mid = 0.5 * (f_lo + f_hi)
        cross = tm[np.nonzero(np.diff(np.sign(sm - mid)))[0]]
        taus = list(cross[:3]) + [tl[0] + q * (tl[-1] - tl[0]) for q in (1 / 3, 2 / 3)]
        i_f, i_v = names.index("f0"), names.index("v")
        v_mid = c * (f_hi - f_lo) / (f_hi + f_lo)  # from the observed extremes (a lower bound)
        for tau in taus:
            for consistent in (False, True):
                alt = p0.copy()
                if consistent:  # rest frequency and speed that match this CPA time
                    alt[i_f], alt[i_v] = mid, max(v_mid, lo[i_v] + 1.0)
                alt[i_t] = tau - alt[i_d] / c
                starts.append(alt)
        # Pick by median |residual|: when sigma understates a wandering tone the
        # robust loss sees most points as outliers and cannot rank the minima.
        best = lmeds(res)
        for alt in starts:
            r_alt = solve(alt)
            m = lmeds(r_alt)
            if m < best * (1.0 - 1e-6):
                res, best = r_alt, m
    r = resid(res.x)
    n_par = len(names)

    # Covariance from the *unmodified* Jacobian on inliers. ``res.jac`` is scaled
    # by the robust loss, which inflates the errors even for clean Gaussian data.
    robust = 1.4826 * float(np.median(np.abs(r)))
    inl = np.abs(r) < max(5.0 * robust, 3.0)  # never reject within 3 stated sigma
    J = _central_jacobian(resid, res.x)[inl]
    cov = np.linalg.pinv(J.T @ J)
    n_inl = int(np.count_nonzero(inl))
    # Goodness of fit on inliers: the robust loss already ignores the rest, and
    # a few wrong-peak assignments at tone crossings should not fail the grade.
    chi2 = float(np.sum(r[inl] ** 2))
    dof = max(n_inl - n_par, 0)
    if not absolute_sigma and dof > 0:
        cov = cov * chi2 / dof
    corr = correlation_factor(r[inl])  # t is sorted, so r is a time series
    cov = cov * corr

    fitted = dict(zip(names, res.x, strict=True))
    pinned = False
    for nme in ("v", "d"):
        if nme in fixed:
            continue
        i = names.index(nme)
        pinned |= bool(res.x[i] <= lo[i] * 1.005 or res.x[i] >= hi[i] * 0.995)
    # Close to the CPA-distance floor the fit is stuck rather than measuring: on
    # real recordings broadband noise fitted as d = 20-22 m with zero variance.
    pinned |= "d" not in fixed and bool(res.x[names.index("d")] < 1.25 * lo[names.index("d")])
    fitted["tau0"] += t_ref
    raw_resid = model(res.x) - f
    ss_tot = float(np.sum((f - f.mean()) ** 2))
    r2 = 1.0 - float(np.sum(raw_resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    return DopplerFit(
        f0=float(fitted["f0"]),
        v=float(fitted["v"]),
        d=float(fitted["d"]),
        tau0=float(fitted["tau0"]),
        f0_rate=float(fitted.get("f0_rate", 0.0)),
        c=c,
        param_names=names,
        cov=cov,
        chi2=chi2,
        dof=dof,
        r2=r2,
        t_range=(float(t[0]), float(t[-1])),
        success=bool(res.success),
        message=str(res.message),
        residuals=raw_resid,
        sigma=np.asarray(sig, dtype=float),
        outlier_fraction=1.0 - n_inl / t.size,
        corr_factor=corr,
        at_bound=pinned,
    )


def bootstrap_fit(
    t: ArrayLike,
    f: ArrayLike,
    fit: DopplerFit,
    k: ArrayLike | None = None,
    n_boot: int = 200,
    rng: np.random.Generator | int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Residual bootstrap around ``fit``. Returns ``(samples, cov)``.

    ``samples`` has shape (n_ok, n_params) in ``fit.param_names`` order; fits
    that fail to converge are dropped.
    """
    rng = np.random.default_rng(rng)
    t, f, k, _ = _prepare(t, f, k, None)
    model = fit.predict(t, k)
    resid = f - model
    x0 = {n: getattr(fit, n) for n in fit.param_names}
    out = []
    for _ in range(n_boot):
        fb = model + rng.choice(resid, size=resid.size, replace=True)
        b = fit_doppler(
            t, fb, k, fit.sigma, c=fit.c, fit_rate="f0_rate" in fit.param_names, x0=x0
        )
        if b.success:
            out.append(b.params())
    samples = np.asarray(out)
    return samples, np.cov(samples, rowvar=False)
