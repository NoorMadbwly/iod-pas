"""
================================================================================
 gauss.py -- Gauss angles-only initial orbit determination
================================================================================
Classical Gauss method: three line-of-sight observations from known observer
positions -> three position vectors (r1, r2, r3), then a velocity at the middle
epoch (via Gibbs / Herrick-Gibbs downstream).

This implementation adds two things beyond a textbook transcription:
  1. an iterative refinement loop that updates the Lagrange f/g coefficients
     with the recovered middle velocity until the slant ranges converge,
  2. a physical-plausibility filter on the eighth-order polynomial roots plus an
     optional fourth-observation residual test to break a genuine root ambiguity.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH_EQ
from .propagate import propagate_two_body


class GaussRootAmbiguityError(Exception):
    """Raised when more than one physically plausible root survives every
    filter and no fourth observation is available to break the tie."""

    def __init__(self, candidates):
        self.candidates = candidates
        super().__init__(
            f"Gauss produced {len(candidates)} physically plausible roots; "
            f"a fourth observation (or prior knowledge) is required to resolve "
            f"the ambiguity.")


def _real_positive_roots(a: float, b: float, c: float) -> list[float]:
    """Real positive roots of  x^8 + a*x^6 + b*x^3 + c = 0."""
    coeffs = np.zeros(9)
    coeffs[0] = 1.0     # x^8
    coeffs[2] = a       # x^6
    coeffs[5] = b       # x^3
    coeffs[8] = c       # x^0
    roots = np.roots(coeffs)
    real_pos = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    return sorted(set(round(r, 6) for r in real_pos))


def _d_matrix(los, r_site):
    """Return (D0, D) where D[i, j] = R_i . (cross product p_j)."""
    l1, l2, l3 = los
    p1 = np.cross(l2, l3)
    p2 = np.cross(l1, l3)
    p3 = np.cross(l1, l2)
    d0 = np.dot(l1, p1)
    p = [p1, p2, p3]
    d = np.array([[np.dot(r_site[i], p[j]) for j in range(3)] for i in range(3)])
    return d0, d


def _slant_ranges(c1, c3, d0, d):
    """Slant ranges (rho1, rho2, rho3) for given Lagrange coefficients c1, c3.

    Derived from  c1*r1 - r2 + c3*r3 = 0  with  r_i = R_i + rho_i * L_i,
    solved with Cramer's rule over the line-of-sight basis.
    """
    rho1 = (-c1 * d[0, 0] + d[1, 0] - c3 * d[2, 0]) / (c1 * d0)
    rho2 = (-c1 * d[0, 1] + d[1, 1] - c3 * d[2, 1]) / d0
    rho3 = (-c1 * d[0, 2] + d[1, 2] - c3 * d[2, 2]) / (c3 * d0)
    return rho1, rho2, rho3


def gauss_method(los_vectors, r_site_eci, times_sec,
                 extra_check_point: dict | None = None,
                 refine: bool = True, max_iter: int = 30) -> dict:
    """Run Gauss's method.

    Parameters
    ----------
    los_vectors : [L1, L2, L3]  ECI line-of-sight unit vectors.
    r_site_eci  : [R1, R2, R3]  observer positions in ECI at each epoch (km).
    times_sec   : [t1, t2, t3]  observation times (seconds, any common origin).
    extra_check_point : optional {"los", "r_site", "t"} fourth observation used
        only to break a root ambiguity.
    refine : iterate the f/g coefficients to convergence (recommended).

    Returns a dict with r1, r2, r3, v2 (if refined), the middle-range magnitude
    and diagnostics.
    """
    l1, l2, l3 = [np.asarray(x, float) for x in los_vectors]
    r1s, r2s, r3s = [np.asarray(x, float) for x in r_site_eci]
    t1, t2, t3 = times_sec

    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    d0, d = _d_matrix([l1, l2, l3], [r1s, r2s, r3s])
    if abs(d0) < 1e-12:
        raise ValueError(
            "D0 ~ 0: the three lines of sight are nearly coplanar/collinear; "
            "choose observations with wider angular separation.")

    # First-order coefficients (independent of the unknown r2).
    a_coef_lin = (1.0 / d0) * (-d[0, 1] * (tau3 / tau) + d[1, 1]
                              + d[2, 1] * (tau1 / tau))
    b_coef_lin = (1.0 / (6.0 * d0)) * (
        d[0, 1] * (tau3 ** 2 - tau ** 2) * (tau3 / tau)
        + d[2, 1] * (tau ** 2 - tau1 ** 2) * (tau1 / tau))

    e = np.dot(r2s, l2)
    r2s_sq = np.dot(r2s, r2s)

    a = -(a_coef_lin ** 2 + 2.0 * a_coef_lin * e + r2s_sq)
    b = -2.0 * MU_EARTH * b_coef_lin * (a_coef_lin + e)
    c = -(MU_EARTH ** 2) * (b_coef_lin ** 2)

    all_roots = _real_positive_roots(a, b, c)

    min_r = R_EARTH_EQ + 150.0        # lowest sensible orbital radius
    max_r = 500000.0                  # beyond geostationary-plus margin
    plausible = [r for r in all_roots if min_r < r < max_r]

    diagnostics = {"all_real_positive_roots": all_roots,
                   "plausible_roots": plausible}

    if not plausible:
        raise ValueError(
            f"No physically plausible root among {all_roots}. Check the site "
            f"positions, the line-of-sight vectors and the observation times.")

    if len(plausible) == 1:
        r2_mag = plausible[0]
    elif extra_check_point is not None:
        r2_mag = _resolve_root_with_extra_point(
            plausible, [l1, l2, l3], [r1s, r2s, r3s],
            [t1, t2, t3], d0, d, extra_check_point)
    else:
        raise GaussRootAmbiguityError(plausible)

    # Initial Lagrange coefficients from the truncated series.
    u = MU_EARTH / r2_mag ** 3
    c1 = (tau3 / tau) * (1.0 + u * (tau ** 2 - tau3 ** 2) / 6.0)
    c3 = -(tau1 / tau) * (1.0 + u * (tau ** 2 - tau1 ** 2) / 6.0)
    rho1, rho2, rho3 = _slant_ranges(c1, c3, d0, d)

    r1 = r1s + rho1 * l1
    r2 = r2s + rho2 * l2
    r3 = r3s + rho3 * l3
    v2 = None
    n_iter = 0

    if refine:
        from .gibbs import compute_velocity_auto
        from .propagate import lagrange_coefficients

        def los_residual(rm, vm):
            """How well a two-body orbit through (rm, vm) reproduces the
            observed lines of sight at t1 and t3 (km, perpendicular miss)."""
            if vm is None or not np.all(np.isfinite(rm)) or not np.all(np.isfinite(vm)):
                return np.inf
            total = 0.0
            with np.errstate(all="ignore"):
                for l_hat, r_site, dt in ((l1, r1s, tau1), (l3, r3s, tau3)):
                    r_pred, _ = propagate_two_body(rm, vm, dt)
                    if not np.all(np.isfinite(r_pred)):
                        return np.inf
                    w = r_pred - r_site
                    total += np.linalg.norm(w - np.dot(w, l_hat) * l_hat)
            return total

        # Seed with the unrefined solution and only accept improvements.
        try:
            v2 = compute_velocity_auto(r1, r2, r3, tau1, 0.0, tau3)["v2"]
        except Exception:
            v2 = None
        best_r2, best_v2 = r2.copy(), (v2.copy() if v2 is not None else None)
        best_res = los_residual(r2, v2) if v2 is not None else np.inf
        best_triplet = (r1.copy(), r2.copy(), r3.copy())

        prev_rho2 = rho2
        for n_iter in range(1, max_iter + 1):
            if v2 is None:
                break
            f1, g1 = lagrange_coefficients(r2, v2, tau1)
            f3, g3 = lagrange_coefficients(r2, v2, tau3)
            denom = f1 * g3 - f3 * g1
            if abs(denom) < 1e-15:
                break
            c1, c3 = g3 / denom, -g1 / denom
            rho1, rho2, rho3 = _slant_ranges(c1, c3, d0, d)
            r1 = r1s + rho1 * l1
            r2 = r2s + rho2 * l2
            r3 = r3s + rho3 * l3
            try:
                v2 = compute_velocity_auto(r1, r2, r3, tau1, 0.0, tau3)["v2"]
            except Exception:
                break
            res = los_residual(r2, v2)
            if res < best_res:
                best_res = res
                best_r2, best_v2 = r2.copy(), v2.copy()
                best_triplet = (r1.copy(), r2.copy(), r3.copy())
            if abs(rho2 - prev_rho2) < 1e-9:
                break
            prev_rho2 = rho2

        r1, r2, r3 = best_triplet
        v2 = best_v2
        diagnostics["los_residual_km"] = float(best_res)

    diagnostics["iterations"] = n_iter
    return {"r1": r1, "r2": r2, "r3": r3, "v2": v2,
            "r2_magnitude": float(np.linalg.norm(r2)),
            "diagnostics": diagnostics}


def _resolve_root_with_extra_point(plausible, los, r_site, times, d0, d,
                                   extra) -> float:
    """Break a root ambiguity by propagating each candidate to a fourth
    observation epoch and keeping the one whose predicted position lies closest
    to the fourth line of sight."""
    from .gibbs import gibbs_velocity
    t1, t2, t3 = times
    tau1, tau3 = t1 - t2, t3 - t2
    best_root, best_residual = None, np.inf

    for r2_mag in plausible:
        u = MU_EARTH / r2_mag ** 3
        c1 = (tau3 / (tau3 - tau1)) * (1.0 + u * ((tau3 - tau1) ** 2 - tau3 ** 2) / 6.0)
        c3 = -(tau1 / (tau3 - tau1)) * (1.0 + u * ((tau3 - tau1) ** 2 - tau1 ** 2) / 6.0)
        rho1, rho2, rho3 = _slant_ranges(c1, c3, d0, d)
        r1 = r_site[0] + rho1 * los[0]
        r2 = r_site[1] + rho2 * los[1]
        r3 = r_site[2] + rho3 * los[2]
        try:
            v2 = gibbs_velocity(r1, r2, r3)
        except Exception:
            continue
        dt = extra["t"] - t2
        r_pred, _ = propagate_two_body(r2, v2, dt)
        w = r_pred - extra["r_site"]
        l_hat = extra["los"] / np.linalg.norm(extra["los"])
        residual = np.linalg.norm(w - np.dot(w, l_hat) * l_hat)
        if residual < best_residual:
            best_residual, best_root = residual, r2_mag

    if best_root is None:
        raise GaussRootAmbiguityError(plausible)
    return best_root
