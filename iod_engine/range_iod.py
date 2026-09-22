"""
================================================================================
 range_iod.py -- Range and range-rate initial orbit determination
================================================================================
Range and range-rate observations do not yield a closed-form state the way three
angles or three positions do: a single scalar range per epoch under-determines a
6-D state. The standard treatment is a batch least-squares estimate ("differential
correction") of the epoch state vector that best reproduces every measurement
under two-body dynamics.

This module provides one unified estimator that handles both required modes:
  * Range + Range-Rate  -- both residual types active (well observable).
  * Range-Only          -- range residuals only (observability depends on the
                           number of stations, arc length and geometry, exactly
                           as the specification notes).

A Levenberg-Marquardt trust-region damps the Gauss-Newton step so the solver
stays stable even when the geometry is poor. A linear trilateration bootstrap
supplies the initial guess when three or more stations see the object.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH_EQ
from .propagate import propagate_two_body


def _predicted_measurements(state0, t0, obs, use_range_rate):
    """Predict range (and range-rate) for every observation given epoch state."""
    r0, v0 = state0[:3], state0[3:]
    preds = []
    for o in obs:
        r, v = propagate_two_body(r0, v0, o["t"] - t0)
        rel_r = r - o["station_pos"]
        rng = np.linalg.norm(rel_r)
        preds.append(rng)
        if use_range_rate:
            rel_v = v - o.get("station_vel", np.zeros(3))
            preds.append(np.dot(rel_r, rel_v) / rng)
    return np.array(preds)


def _observation_vector(obs, use_range_rate):
    vec = []
    for o in obs:
        vec.append(o["range"])
        if use_range_rate:
            vec.append(o["range_rate"])
    return np.array(vec)


def _trilaterate(obs):
    """Linear least-squares position from three or more ranges near one epoch."""
    epoch0 = obs[0]["t"]
    cluster = [o for o in obs if abs(o["t"] - epoch0) < 1.0]
    stations = {tuple(np.round(o["station_pos"], 3)) for o in cluster}
    if len(stations) < 3:
        return None
    ref = cluster[0]
    a_rows, b_rows = [], []
    for o in cluster[1:]:
        a_rows.append(2.0 * (o["station_pos"] - ref["station_pos"]))
        b_rows.append(np.dot(o["station_pos"], o["station_pos"])
                      - np.dot(ref["station_pos"], ref["station_pos"])
                      - (o["range"] ** 2 - ref["range"] ** 2))
    a = np.array(a_rows)
    b = np.array(b_rows)
    pos, *_ = np.linalg.lstsq(a, b, rcond=None)
    return pos


def _initial_guess(obs, provided):
    if provided is not None:
        return np.asarray(provided, float)

    pos = _trilaterate(obs)
    if pos is None:
        # single-station bootstrap: place the object above the station at the
        # first measured range, with a circular speed perpendicular to zenith.
        o = obs[0]
        r_stn = o["station_pos"]
        zenith = r_stn / np.linalg.norm(r_stn)
        pos = r_stn + o["range"] * zenith
    r_mag = np.linalg.norm(pos)
    speed = np.sqrt(MU_EARTH / r_mag)
    # velocity guess perpendicular to the radius, in the equatorial-ish plane
    radial = pos / r_mag
    tangent = np.cross([0.0, 0.0, 1.0], radial)
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    vel = speed * tangent
    return np.concatenate([pos, vel])


def range_iod(observations, initial_guess=None, use_range_rate=True,
              max_iter: int = 200, tol: float = 1e-10) -> dict:
    """Estimate the epoch state vector from range / range-rate observations.

    Parameters
    ----------
    observations : list of dicts, each with keys
        ``t`` (seconds), ``station_pos`` (ECI km), ``range`` (km) and, when
        ``use_range_rate`` is True, ``station_vel`` (ECI km/s) and
        ``range_rate`` (km/s).
    initial_guess : optional 6-vector [x,y,z,vx,vy,vz] at the first epoch.
    use_range_rate : include range-rate residuals (Range+Range-Rate mode) or not
        (Range-Only mode).

    Returns the estimated epoch state, RMS residual, iteration count and
    convergence flag. Uses a trust-region reflective least-squares solver when
    SciPy is available, falling back to a damped Gauss-Newton otherwise.
    """
    obs = sorted(observations, key=lambda o: o["t"])
    t0 = obs[0]["t"]
    x0 = _initial_guess(obs, initial_guess)
    y = _observation_vector(obs, use_range_rate)

    def residuals(x):
        return _predicted_measurements(x, t0, obs, use_range_rate) - y

    converged = False
    iterations = 0
    try:
        from scipy.optimize import least_squares
        sol = least_squares(residuals, x0, method="trf",
                            x_scale="jac", max_nfev=max_iter * 8,
                            ftol=tol, xtol=tol, gtol=tol)
        x = sol.x
        converged = bool(sol.success)
        iterations = int(sol.nfev)
    except ImportError:
        x, converged, iterations = _damped_gauss_newton(
            residuals, x0, max_iter, tol)

    resid = residuals(x)
    rms = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else None

    return {
        "epoch_state": x,
        "r0": x[:3], "v0": x[3:],
        "reference_time_sec": t0,
        "rms_residual": rms,
        "iterations": iterations,
        "converged": converged,
        "mode": "range+range-rate" if use_range_rate else "range-only",
        "n_observations": len(obs),
    }


def _damped_gauss_newton(residuals, x0, max_iter, tol):
    """Levenberg-Marquardt fallback used only when SciPy is unavailable."""
    x = x0.copy()
    lam = 1e-3
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        r = residuals(x)
        cost = float(r @ r)
        n = len(r)
        jac = np.zeros((n, 6))
        for k in range(6):
            step = max(1e-3, abs(x[k]) * 1e-6)
            xp = x.copy(); xp[k] += step
            jac[:, k] = (residuals(xp) - r) / step
        jtj, jtr = jac.T @ jac, -jac.T @ r
        try:
            dx = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj)), jtr)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(jtj + lam * np.eye(6), jtr, rcond=None)[0]
        r_new = residuals(x + dx)
        if float(r_new @ r_new) < cost:
            x = x + dx
            lam = max(lam * 0.5, 1e-9)
            if abs(prev - cost) < tol * max(1.0, cost):
                return x, True, it
            prev = cost
        else:
            lam = min(lam * 4.0, 1e8)
    return x, False, it
