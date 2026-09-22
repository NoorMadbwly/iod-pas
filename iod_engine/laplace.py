"""
================================================================================
 laplace.py -- Laplace angles-only initial orbit determination
================================================================================
Where Gauss works from three discrete lines of sight, Laplace works from the
line-of-sight vector and its first and second time derivatives at a single
central epoch. The derivatives are obtained by Lagrange interpolation of three
(or more) angle observations, and the observer's own velocity and acceleration
(from Earth rotation) enter the solution.

The two-body condition r'' = -mu r / r^3, applied to r = R + rho * L, reduces to
an eighth-order polynomial in the geocentric radius, exactly as in Gauss.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH_EQ


def _los_derivatives(vals, times, centre_index):
    """Value, first and second derivative of the LOS at the central epoch.

    For three samples this is the classical quadratic Lagrange estimate; for
    more samples each Cartesian component is least-squares fitted with a
    polynomial of degree ``min(N-1, 4)`` and differentiated analytically at the
    central time. Extra observations sharply improve the derivative accuracy,
    which is the dominant Laplace error source.
    """
    vals = np.asarray(vals, dtype=float)          # (N, 3)
    t = np.asarray(times, dtype=float)
    tc = t[centre_index]
    dt = t - tc                                    # centre at the epoch
    n = len(t)
    degree = min(n - 1, 4)

    l = vals[centre_index]
    ldot = np.zeros(3)
    lddot = np.zeros(3)
    for axis in range(3):
        coeffs = np.polyfit(dt, vals[:, axis], degree)   # highest power first
        d1 = np.polyder(coeffs, 1)
        d2 = np.polyder(coeffs, 2)
        l[axis] = np.polyval(coeffs, 0.0)
        ldot[axis] = np.polyval(d1, 0.0)
        lddot[axis] = np.polyval(d2, 0.0)
    return l, ldot, lddot


def _real_positive_roots(alpha6, beta3, gamma0):
    coeffs = np.zeros(9)
    coeffs[0] = 1.0
    coeffs[2] = alpha6
    coeffs[5] = beta3
    coeffs[8] = gamma0
    roots = np.roots(coeffs)
    return sorted(set(round(r.real, 6) for r in roots
                      if abs(r.imag) < 1e-6 and r.real > 0))


def laplace_method(los_vectors, site_states, times_sec) -> dict:
    """Run Laplace's method.

    Parameters
    ----------
    los_vectors : [L1, ..., LN] ECI line-of-sight unit vectors (N >= 3).
    site_states : [(R, V, A), ...] observer position/velocity/acceleration in
        ECI at each epoch. Only the central state is used directly; all epochs
        fix the LOS derivatives.
    times_sec   : [t1, ..., tN].

    Returns a dict with the state (r, v) at the central epoch and diagnostics.
    """
    los = [np.asarray(x, float) for x in los_vectors]
    t = list(times_sec)
    if len(los) < 3:
        raise ValueError("Laplace requires at least three observations.")

    centre = len(los) // 2
    l, ldot, lddot = _los_derivatives(los, t, centre)

    r_site, v_site, a_site = (np.asarray(x, float) for x in site_states[centre])

    # Determinants (scalar triple products).
    delta = np.dot(l, np.cross(ldot, lddot))
    if abs(delta) < 1e-15:
        raise ValueError(
            "Laplace determinant ~ 0: the line-of-sight derivatives are "
            "degenerate; use observations with more curvature/spacing.")
    big_d = 2.0 * delta

    d1 = np.dot(l, np.cross(ldot, a_site))
    d2 = np.dot(l, np.cross(ldot, r_site))
    d3 = np.dot(l, np.cross(lddot, a_site))
    d4 = np.dot(l, np.cross(lddot, r_site))

    a2 = -2.0 * d1 / big_d
    b2 = -2.0 * MU_EARTH * d2 / big_d

    c = np.dot(r_site, l)
    r_site_sq = np.dot(r_site, r_site)

    alpha6 = -(a2 ** 2 + 2.0 * c * a2 + r_site_sq)
    beta3 = -2.0 * b2 * (a2 + c)
    gamma0 = -(b2 ** 2)

    all_roots = _real_positive_roots(alpha6, beta3, gamma0)
    min_r = R_EARTH_EQ + 150.0
    max_r = 500000.0
    plausible = [r for r in all_roots if min_r < r < max_r]

    if not plausible:
        raise ValueError(
            f"No physically plausible geocentric radius among {all_roots}.")

    # If several survive, keep the one giving the smallest positive slant range.
    best = None
    for r_geo in plausible:
        rho = a2 + b2 / r_geo ** 3
        if rho <= 0:
            continue
        if best is None or rho < best[1]:
            best = (r_geo, rho)
    if best is None:
        # fall back to the smallest radius even if slant range is marginal
        r_geo = plausible[0]
        rho = a2 + b2 / r_geo ** 3
    else:
        r_geo, rho = best

    rho_dot = (d3 + (MU_EARTH / r_geo ** 3) * d4) / big_d

    r_vec = r_site + rho * l
    v_vec = v_site + rho_dot * l + rho * ldot

    return {"r2": r_vec, "v2": v_vec,
            "slant_range_km": float(rho),
            "slant_range_rate_kms": float(rho_dot),
            "diagnostics": {"all_real_positive_roots": all_roots,
                            "plausible_roots": plausible,
                            "geocentric_radius_km": float(r_geo)}}
