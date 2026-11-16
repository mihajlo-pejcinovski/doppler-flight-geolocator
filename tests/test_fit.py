"""Single-pass fit: recovery, robustness, and covariance calibration."""

import numpy as np
import pytest

from dopplergeo.core import (
    SPEED_OF_SOUND_15C,
    bootstrap_fit,
    doppler_frequency,
    fit_doppler,
    initial_guess,
)

C = SPEED_OF_SOUND_15C
TRUE = {"f0": 180.0, "v": 140.0, "d": 1800.0, "tau0": 20.0}


def make_track(n=600, span=(-60, 90), sigma=0.0, rng=0, k=1.0, f0_rate=0.0, **over):
    p = {**TRUE, **over}
    t = np.linspace(*span, n)
    f = doppler_frequency(t, p["f0"], p["v"], p["d"], p["tau0"], C, k, f0_rate)
    if sigma:
        f = f + np.random.default_rng(rng).normal(0, sigma, n)
    return t, f


def test_initial_guess_is_in_the_right_basin():
    t, f = make_track()
    g = initial_guess(t, f)
    assert g["f0"] == pytest.approx(TRUE["f0"], rel=0.02)
    assert g["v"] == pytest.approx(TRUE["v"], rel=0.15)
    assert 0.5 < g["d"] / TRUE["d"] < 2.0
    assert abs(g["tau0"] - TRUE["tau0"]) < 5.0


def test_noiseless_recovery():
    t, f = make_track()
    fit = fit_doppler(t, f, sigma=0.01)
    assert fit.success
    assert fit.f0 == pytest.approx(TRUE["f0"], rel=1e-6)
    assert fit.v == pytest.approx(TRUE["v"], rel=1e-5)
    assert fit.d == pytest.approx(TRUE["d"], rel=1e-5)
    assert fit.tau0 == pytest.approx(TRUE["tau0"], abs=1e-4)
    assert fit.grade == "A"


def test_noisy_recovery_without_sigma_given():
    t, f = make_track(sigma=0.3, rng=1)
    fit = fit_doppler(t, f)
    for name in ("f0", "v", "d", "tau0"):
        assert abs(getattr(fit, name) - TRUE[name]) < 4 * fit.stderr(name), name
    assert fit.chi2_red == pytest.approx(1.0, abs=0.25)


def test_utc_epoch_times():
    tau0 = 1.79e9
    t, f = make_track(tau0=tau0, span=(tau0 - 60, tau0 + 90), sigma=0.2)
    fit = fit_doppler(t, f)
    assert fit.tau0 == pytest.approx(tau0, abs=0.5)
    assert fit.t_cpa == pytest.approx(tau0 + TRUE["d"] / C, abs=0.5)


def test_harmonics_fitted_jointly_share_geometry():
    t1, f1 = make_track(n=300, sigma=0.3, rng=2, k=1)
    t2, f2 = make_track(n=300, sigma=0.3, rng=3, k=2)
    t3, f3 = make_track(n=300, sigma=0.3, rng=4, k=3)
    t = np.concatenate([t1, t2, t3])
    f = np.concatenate([f1, f2, f3])
    k = np.concatenate([np.full(300, 1), np.full(300, 2), np.full(300, 3)])
    joint = fit_doppler(t, f, k=k)
    single = fit_doppler(t1, f1)
    assert joint.v == pytest.approx(TRUE["v"], rel=0.01)
    assert joint.stderr("v") < single.stderr("v")


def test_f0_drift():
    t, f = make_track(sigma=0.1, rng=5, f0_rate=0.05)
    fit = fit_doppler(t, f, fit_rate=True)
    assert fit.f0_rate == pytest.approx(0.05, abs=0.01)
    assert fit.d == pytest.approx(TRUE["d"], rel=0.05)


def test_robust_to_outliers():
    t, f = make_track(sigma=0.3, rng=6)
    rng = np.random.default_rng(7)
    bad = rng.choice(t.size, size=t.size // 20, replace=False)
    f[bad] += rng.uniform(-40, 40, bad.size)
    fit = fit_doppler(t, f)
    assert fit.v == pytest.approx(TRUE["v"], rel=0.02)
    assert fit.d == pytest.approx(TRUE["d"], rel=0.05)


def test_no_cpa_in_window_is_not_grade_a():
    t, f = make_track(span=(-60, 0), sigma=0.3, rng=8)  # approach only
    fit = fit_doppler(t, f)
    assert not fit.cpa_covered
    assert fit.grade != "A"


def test_covariance_is_calibrated_monte_carlo():
    """Spec M1 gate: truth inside +-1 sigma in ~68% of noisy runs."""
    n_runs, sigma = 500, 0.5
    t, f_true = make_track(n=300)
    rng = np.random.default_rng(42)
    hits = {n: 0 for n in ("f0", "v", "d", "tau0")}
    for _ in range(n_runs):
        fit = fit_doppler(t, f_true + rng.normal(0, sigma, t.size), sigma=sigma)
        for n in hits:
            hits[n] += abs(getattr(fit, n) - TRUE[n]) <= fit.stderr(n)
    for n, h in hits.items():
        assert 0.62 <= h / n_runs <= 0.74, (n, h / n_runs)


def test_bootstrap_agrees_with_covariance():
    t, f = make_track(n=300, sigma=0.5, rng=9)
    fit = fit_doppler(t, f)
    samples, cov = bootstrap_fit(t, f, fit, n_boot=150, rng=10)
    assert samples.shape[0] >= 140
    for i, name in enumerate(fit.param_names):
        ratio = np.sqrt(cov[i, i]) / fit.stderr(name)
        assert 0.67 < ratio < 1.5, (name, ratio)


def test_far_extrapolated_cpa_and_zero_error_fits_are_not_graded_b():
    # Only the tail of a departure: the fit must extrapolate the CPA far before the data.
    t, f = make_track(span=(400, 520), sigma=0.05, rng=3)
    fit = fit_doppler(t, f)
    assert not fit.cpa_near
    assert fit.grade == "C"
    # A real CPA pass with realistic noise stays A/B and is not "degenerate".
    t, f = make_track(sigma=0.2, rng=4)
    fit = fit_doppler(t, f)
    assert fit.cpa_near and fit.grade in ("A", "B")
