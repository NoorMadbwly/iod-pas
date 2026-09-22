"""
================================================================================
 validation.py -- Physical sanity checks and out-of-sample residual QC
================================================================================
Every computed state passes a physical-plausibility gate before it is trusted or
stored, and (when spare observations exist) an out-of-sample residual check that
propagates the solution to epochs it was not fitted against.
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH_EQ
from .elements import rv_to_elements
from .propagate import propagate_two_body


class UnphysicalResultError(Exception):
    pass


def sanity_check_result(r2: np.ndarray, v2: np.ndarray) -> dict:
    """Reject states that cannot correspond to a real bound Earth orbit.

    Returns the full element set on success; raises with a detailed message on
    failure so a bad root or bad input never propagates silently.
    """
    elements = rv_to_elements(r2, v2)
    issues = []

    e = elements["eccentricity"]
    if not (0.0 <= e < 1.0):
        issues.append(
            f"eccentricity e={e:.4f} is not that of a closed orbit (expected "
            f"0 <= e < 1); likely a bad Gauss root or bad input.")

    peri = elements["perigee_altitude_km"]
    if peri is not None and peri < 100.0:
        issues.append(
            f"perigee altitude {peri:.1f} km places the orbit inside the "
            f"atmosphere/Earth; almost certainly an incorrect result.")

    r_mag = np.linalg.norm(r2)
    if r_mag < R_EARTH_EQ:
        issues.append(
            f"|r2| = {r_mag:.1f} km is below Earth's radius; physically "
            f"impossible.")

    if issues:
        raise UnphysicalResultError(
            "Result failed the physical-plausibility gate:\n"
            + "\n".join(f"  - {i}" for i in issues))
    return elements


def residual_self_check(r2, v2, t2, extra_true_states) -> dict:
    """Propagate the solution to unused epochs and compare with truth.

    ``extra_true_states`` is a list of {"t", "r_true"} not used in the fit.
    Provides a genuine out-of-sample accuracy figure.
    """
    errors = []
    for extra in extra_true_states:
        r_pred, _ = propagate_two_body(r2, v2, extra["t"] - t2)
        if "r_true" in extra:
            errors.append(float(np.linalg.norm(r_pred - extra["r_true"])))

    if not errors:
        return {"n_checked": 0, "mean_error_km": None, "max_error_km": None}
    return {"n_checked": len(errors),
            "mean_error_km": float(np.mean(errors)),
            "max_error_km": float(np.max(errors)),
            "all_errors_km": errors}
