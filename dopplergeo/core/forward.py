"""Compiled forward model: retarded-time solve + Doppler factor per point.

One numba kernel evaluates, for every (observer, reception time) pair, the
Newton solve for the emission time against a ``SampledTrajectory`` (cubic
Hermite nodes) and the log Doppler factor, in still air or in moving air
(uniform wind, ISA-lapse path-mean speed of sound). The NumPy versions in
``solver`` built ~15 full-size temporaries per Newton iteration and were
memory-bound; here each point stays in registers and the points run in parallel.
The NumPy code is kept as the reference the tests compare against.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .atmosphere import _A, ISA_LAPSE_RATE

_TOL = 1e-8


@numba.njit(cache=True, nogil=True, inline="always")
def _herm(P, V, ok, t0, dt, last, tau):
    """Hermite position/velocity at tau, linear extrapolation past the ends
    (flagged invalid) -- the same arithmetic as SampledTrajectory.eval."""
    x = (tau - t0) / dt
    fi = math.floor(x) if math.isfinite(x) else 0.0
    i = int(min(max(fi, 0.0), float(last)))
    u = x - i
    uc = min(max(u, 0.0), 1.0)
    u2 = uc * uc
    u3 = u2 * uc
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + uc
    h01 = 3 * u2 - 2 * u3
    h11 = u3 - u2
    d00 = 6 * u2 - 6 * uc
    d10 = 3 * u2 - 4 * uc + 1
    d01 = 6 * uc - 6 * u2
    d11 = 3 * u2 - 2 * uc
    beyond = (u - uc) * dt
    px = py = pz = vx = vy = vz = 0.0
    for k in range(3):
        p0 = P[i, k]
        p1 = P[i + 1, k]
        v0 = V[i, k] * dt
        v1 = V[i + 1, k] * dt
        vel = (d00 * p0 + d10 * v0 + d01 * p1 + d11 * v1) / dt
        pos = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1 + beyond * vel
        if k == 0:
            px, vx = pos, vel
        elif k == 1:
            py, vy = pos, vel
        else:
            pz, vz = pos, vel
    valid = (u >= -1e-9) and (u <= 1 + 1e-9) and ok[i] and ok[i + 1]
    return px, py, pz, vx, vy, vz, valid


@numba.njit(cache=True, nogil=True, inline="always")
def _c_path(dh, t0k, lapse, c_const):
    """Travel-time mean speed of sound over a ray climbing dh (Atmosphere.c_path)."""
    if c_const > 0.0:
        return c_const
    if not lapse:
        return _A * math.sqrt(t0k)
    L = ISA_LAPSE_RATE
    dh = min(max(dh, -1000.0), 11_000.0)
    if abs(dh) < 1.0:
        return _A * math.sqrt(t0k - 0.5 * L * dh)
    return _A * L * dh / (2.0 * (math.sqrt(t0k) - math.sqrt(t0k - L * dh)))


@numba.njit(cache=True, nogil=True, inline="always")
def _point(j, P, V, ok, t0, dt, last, obs, t, wx, wy, wz, t0k, lapse, c_const, still, iters, logd, valid, tau_out):
    """One point of the kernel (see _kernel_parallel for the model)."""
    ox, oy, oz = obs[j, 0], obs[j, 1], obs[j, 2]
    tj = t[j]
    px, py, pz, vx, vy, vz, _ = _herm(P, V, ok, t0, dt, last, tj)
    c = _c_path(pz - oz, t0k, lapse, c_const)
    tau = tj - math.sqrt((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2) / c
    for _ in range(iters):
        px, py, pz, vx, vy, vz, _ = _herm(P, V, ok, t0, dt, last, tau)
        if still:
            rx, ry, rz = px - ox, py - oy, pz - oz
            dist = math.sqrt(rx * rx + ry * ry + rz * rz)
            vr = (vx * rx + vy * ry + vz * rz) / dist
            step = (tau + dist / c - tj) / (1.0 + vr / c)
        else:
            c = _c_path(pz - oz, t0k, lapse, c_const)
            s = tj - tau
            qx, qy, qz = ox - px - wx * s, oy - py - wy * s, oz - pz - wz * s
            qn = math.sqrt(qx * qx + qy * qy + qz * qz)
            inv = 1.0 / max(qn, 1e-9)
            gp = c + (qx * (wx - vx) + qy * (wy - vy) + qz * (wz - vz)) * inv
            step = (qn - c * s) / max(gp, 0.05 * c)
        tau -= step
        if not abs(step) >= _TOL:  # converged (or NaN: leave it, flagged below)
            break
    px, py, pz, vx, vy, vz, ok_j = _herm(P, V, ok, t0, dt, last, tau)
    if still:
        rx, ry, rz = px - ox, py - oy, pz - oz
        vr = (vx * rx + vy * ry + vz * rz) / math.sqrt(rx * rx + ry * ry + rz * rz)
        logd[j] = -math.log1p(min(max(vr / c, -0.95), 0.95))
        valid[j] = ok_j and math.isfinite(vr)
    else:
        c = _c_path(pz - oz, t0k, lapse, c_const)
        s = tj - tau
        qx, qy, qz = ox - px - wx * s, oy - py - wy * s, oz - pz - wz * s
        inv = 1.0 / max(math.sqrt(qx * qx + qy * qy + qz * qz), 1e-9)
        num = c + (qx * wx + qy * wy + qz * wz) * inv
        den = c + (qx * (wx - vx) + qy * (wy - vy) + qz * (wz - vz)) * inv
        logd[j] = math.log(num / max(den, 0.05 * c))
        valid[j] = ok_j and math.isfinite(den)
    tau_out[j] = tau


@numba.njit(cache=True, nogil=True, parallel=True)
def _kernel_parallel(P, V, ok, t0, dt, obs, t, wind, t0k, lapse, c_const, still, iters, logd, valid, tau_out):
    """For each point j: emission time tau and log Doppler factor.

    still=True: the still-air model at constant c_const (log D = -log1p(v_r/c),
    v_r/c clipped to +-0.95 as the NumPy version does). Otherwise moving air:
    |q| = c (t - tau), q = obs - p(tau) - w (t - tau), f/f0 = (c + q^.w) / (c + q^.(w - v)),
    c the path mean between observer and aircraft heights (c_const > 0 overrides it)."""
    last = P.shape[0] - 2
    for j in numba.prange(t.size):
        _point(j, P, V, ok, t0, dt, last, obs, t, wind[0], wind[1], wind[2], t0k, lapse, c_const, still, iters,
               logd, valid, tau_out)


@numba.njit(cache=True, nogil=True)
def _kernel_serial(P, V, ok, t0, dt, obs, t, wind, t0k, lapse, c_const, still, iters, logd, valid, tau_out):
    """_kernel_parallel without threads: for the refinement's small calls (a few
    thousand points), where starting the thread pool costs more than the work."""
    last = P.shape[0] - 2
    for j in range(t.size):
        _point(j, P, V, ok, t0, dt, last, obs, t, wind[0], wind[1], wind[2], t0k, lapse, c_const, still, iters,
               logd, valid, tau_out)


# Calls with fewer points run serially (measured: a 3000-point call took 3.4 ms in
# NumPy, 6.6 ms with the thread pool on a busy machine).
PARALLEL_MIN_POINTS = 20_000


def log_doppler_fast(traj, obs, t, c: float, atmos=None, iters: int | None = None,
                     parallel: bool | None = None):
    """Compiled ``solver.log_doppler``: same arguments, broadcasting and results.
    ``parallel``: None picks by size; False forces the serial kernel, which callers
    running several evaluations on their own threads must use (numba's workqueue
    threading layer does not allow concurrent parallel regions)."""
    t = np.asarray(t, dtype=float)
    obs = np.asarray(obs, dtype=float)
    shape = np.broadcast_shapes(t.shape, obs.shape[:-1])
    tt = np.ascontiguousarray(np.broadcast_to(t, shape), dtype=np.float64).ravel()
    oo = np.ascontiguousarray(np.broadcast_to(obs, shape + (3,)), dtype=np.float64).reshape(-1, 3)
    n = tt.size
    logd, valid, tau = np.empty(n), np.empty(n, dtype=np.bool_), np.empty(n)
    if atmos is None:
        still, wind, t0k, lapse, c_const = True, np.zeros(3), 288.15, False, float(c)
        it = 10 if iters is None else iters
    else:
        still, wind, t0k, lapse, c_const = False, np.asarray(atmos.wind, dtype=float), atmos.temp_c + 273.15, \
            bool(atmos.lapse), 0.0
        it = 12 if iters is None else iters
    if n:
        use_par = n >= PARALLEL_MIN_POINTS if parallel is None else parallel
        kernel = _kernel_parallel if use_par else _kernel_serial
        kernel(np.ascontiguousarray(traj.P, dtype=np.float64), np.ascontiguousarray(traj.V, dtype=np.float64),
                np.ascontiguousarray(traj.ok, dtype=np.bool_), float(traj.t0), float(traj.dt), oo, tt, wind, float(t0k), lapse,
                c_const, still, int(it), logd, valid, tau)
    return logd.reshape(shape), valid.reshape(shape), tau.reshape(shape)
