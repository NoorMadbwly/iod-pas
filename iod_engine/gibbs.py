"""
================================================================================
 gibbs.py -- Velocity from three position vectors (Gibbs / Herrick-Gibbs)
================================================================================
Given three coplanar position vectors and their times, recover the velocity at
the middle point. Gibbs is purely geometric and suits wide angular spacing;
Herrick-Gibbs is a Taylor expansion and suits closely-spaced points. The unified
``compute_velocity_auto`` picks the appropriate method from the geometry.

Threshold policy (matches the Orekit / Vallado convention):
    separation < 1 deg      -> Herrick-Gibbs
    separation > 5 deg      -> Gibbs
    1 deg .. 5 deg          -> transition zone, prefer Herrick-Gibbs
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH
from .geometry import angular_separation_deg

COPLANAR_THRESHOLD_DEG = 5.0
HERRICK_GIBBS_MAX_DEG = 1.0
GIBBS_MIN_DEG = 5.0


class CoplanarityError(Exception):
    pass


def check_coplanarity(r1, r2, r3, threshold_deg: float = COPLANAR_THRESHOLD_DEG) -> float:
    """Verify the three positions are close to a common orbital plane.

    Returns the out-of-plane deviation in degrees; raises if it exceeds
    ``threshold_deg``.
    """
    n = np.cross(r1, r2)
    n_hat = n / np.linalg.norm(n)
    r3_hat = r3 / np.linalg.norm(r3)
    deviation = 90.0 - np.degrees(np.arccos(
        np.clip(abs(np.dot(n_hat, r3_hat)), -1.0, 1.0)))
    if deviation > threshold_deg:
        raise CoplanarityError(
            f"Positions are not coplanar enough (deviation {deviation:.2f} deg "
            f"> allowed {threshold_deg} deg); result would be unreliable.")
    return deviation


def gibbs_velocity(r1, r2, r3) -> np.ndarray:
    """Classical Gibbs method (best for angular separation > ~5 deg)."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r3 = np.asarray(r3, float)
    m1, m2, m3 = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)

    n = m1 * np.cross(r2, r3) + m2 * np.cross(r3, r1) + m3 * np.cross(r1, r2)
    d = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    s = r1 * (m2 - m3) + r2 * (m3 - m1) + r3 * (m1 - m2)

    n_mag, d_mag = np.linalg.norm(n), np.linalg.norm(d)
    if n_mag < 1e-9 or d_mag < 1e-9:
        raise CoplanarityError("Gibbs N or D vector ~ 0; geometry is singular.")

    return np.sqrt(MU_EARTH / (n_mag * d_mag)) * (np.cross(d, r2) / m2 + s)


def herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3) -> np.ndarray:
    """Herrick-Gibbs method (best for closely-spaced observations)."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    r3 = np.asarray(r3, float)
    dt31 = t3 - t1
    dt21 = t2 - t1
    dt32 = t3 - t2
    m1, m2, m3 = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)

    return (
        -dt32 * (1.0 / (dt21 * dt31) + MU_EARTH / (12.0 * m1 ** 3)) * r1
        + (dt32 - dt21) * (1.0 / (dt21 * dt32) + MU_EARTH / (12.0 * m2 ** 3)) * r2
        + dt21 * (1.0 / (dt32 * dt31) + MU_EARTH / (12.0 * m3 ** 3)) * r3
    )


def compute_velocity_auto(r1, r2, r3, t1, t2, t3) -> dict:
    """Check coplanarity, measure spacing, and pick the appropriate method.

    Falls back from Gibbs to Herrick-Gibbs automatically if Gibbs hits a
    singularity.
    """
    deviation = check_coplanarity(r1, r2, r3)
    sep = min(angular_separation_deg(r1, r2), angular_separation_deg(r2, r3))

    fallback_note = None
    if sep < HERRICK_GIBBS_MAX_DEG:
        method = "Herrick-Gibbs"
        v2 = herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3)
    elif sep > GIBBS_MIN_DEG:
        method = "Gibbs"
        try:
            v2 = gibbs_velocity(r1, r2, r3)
        except CoplanarityError:
            method = "Herrick-Gibbs (fallback)"
            fallback_note = "Gibbs singular; fell back to Herrick-Gibbs."
            v2 = herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3)
    else:
        method = "Herrick-Gibbs (transition zone)"
        v2 = herrick_gibbs_velocity(r1, r2, r3, t1, t2, t3)

    return {"v2": v2, "method_used": method,
            "min_angular_separation_deg": sep,
            "coplanarity_deviation_deg": deviation,
            "fallback_note": fallback_note}
