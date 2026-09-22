"""
================================================================================
 propagate.py -- Two-body propagation (universal variable formulation)
================================================================================
A single Keplerian propagator that works for any conic (elliptic, circular,
parabolic, hyperbolic) without special-case branching. Used for:
  * resolving the Gauss root ambiguity via a residual check,
  * out-of-sample validation of a computed state,
  * modelling predicted range / range-rate measurements in the range solvers,
  * generating truth trajectories for the synthetic test suite.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH


def _stumpff_c2(psi: float) -> float:
    if psi > 1e-6:
        return (1.0 - np.cos(np.sqrt(psi))) / psi
    if psi < -1e-6:
        return (np.cosh(np.sqrt(-psi)) - 1.0) / (-psi)
    return 0.5


def _stumpff_c3(psi: float) -> float:
    if psi > 1e-6:
        s = np.sqrt(psi)
        return (s - np.sin(s)) / s ** 3
    if psi < -1e-6:
        s = np.sqrt(-psi)
        return (np.sinh(s) - s) / s ** 3
    return 1.0 / 6.0


def propagate_two_body(r0: np.ndarray, v0: np.ndarray, dt: float,
                       mu: float = MU_EARTH):
    """Propagate a state (r0, v0) forward by ``dt`` seconds.

    Returns the new (position, velocity) tuple.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if dt == 0.0:
        return r0.copy(), v0.copy()

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    sqrt_mu = np.sqrt(mu)
    sigma0 = np.dot(r0, v0) / sqrt_mu           # = r0 . v0 / sqrt(mu)
    alpha = 2.0 / r0_mag - v0_mag ** 2 / mu     # reciprocal semi-major axis

    chi = sqrt_mu * abs(alpha) * dt             # initial guess

    for _ in range(200):
        psi = chi ** 2 * alpha
        c2, c3 = _stumpff_c2(psi), _stumpff_c3(psi)
        r = (chi ** 2 * c2
             + sigma0 * chi * (1.0 - psi * c3)
             + r0_mag * (1.0 - psi * c2))
        t = (chi ** 3 * c3
             + sigma0 * chi ** 2 * c2
             + r0_mag * chi * (1.0 - psi * c3)) / sqrt_mu
        if abs(r) < 1e-12 or not np.isfinite(r):
            break
        chi_new = chi + (dt - t) * sqrt_mu / r
        if not np.isfinite(chi_new):
            break
        if abs(chi_new - chi) < 1e-9:
            chi = chi_new
            break
        chi = chi_new

    psi = chi ** 2 * alpha
    c2, c3 = _stumpff_c2(psi), _stumpff_c3(psi)

    f = 1.0 - chi ** 2 / r0_mag * c2
    g = dt - chi ** 3 / sqrt_mu * c3
    r_new = f * r0 + g * v0
    r_new_mag = np.linalg.norm(r_new)

    fdot = sqrt_mu / (r_new_mag * r0_mag) * (psi * c3 - 1.0) * chi
    gdot = 1.0 - chi ** 2 / r_new_mag * c2
    v_new = fdot * r0 + gdot * v0
    return r_new, v_new


def lagrange_coefficients(r0: np.ndarray, v0: np.ndarray, dt: float,
                          mu: float = MU_EARTH):
    """Exact two-body Lagrange f, g coefficients over an interval ``dt``.

    Defined by  r(t0+dt) = f * r0 + g * v0.  These are what the Gauss
    refinement needs to converge robustly, in place of the truncated series.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if dt == 0.0:
        return 1.0, 0.0

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    sqrt_mu = np.sqrt(mu)
    sigma0 = np.dot(r0, v0) / sqrt_mu
    alpha = 2.0 / r0_mag - v0_mag ** 2 / mu

    chi = sqrt_mu * abs(alpha) * dt
    for _ in range(200):
        psi = chi ** 2 * alpha
        c2, c3 = _stumpff_c2(psi), _stumpff_c3(psi)
        r = (chi ** 2 * c2 + sigma0 * chi * (1.0 - psi * c3)
             + r0_mag * (1.0 - psi * c2))
        t = (chi ** 3 * c3 + sigma0 * chi ** 2 * c2
             + r0_mag * chi * (1.0 - psi * c3)) / sqrt_mu
        if abs(r) < 1e-12:
            break
        chi_new = chi + (dt - t) * sqrt_mu / r
        if abs(chi_new - chi) < 1e-10:
            chi = chi_new
            break
        chi = chi_new

    psi = chi ** 2 * alpha
    c2, c3 = _stumpff_c2(psi), _stumpff_c3(psi)
    f = 1.0 - chi ** 2 / r0_mag * c2
    g = dt - chi ** 3 / sqrt_mu * c3
    return f, g
