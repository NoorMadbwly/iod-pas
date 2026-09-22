"""
================================================================================
 geometry.py -- Observation geometry quality metrics
================================================================================
Tools for judging how well-conditioned a set of angle observations is, and for
automatically picking the best three-observation subset (triplet) to feed a
three-point method such as Gauss, Gibbs or Herrick-Gibbs.
================================================================================
"""

from __future__ import annotations

import numpy as np
from itertools import combinations


def angular_separation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors, in degrees."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    cos_ang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang))


def compute_gdop(los_vectors: list[np.ndarray]) -> float:
    """Geometric Dilution of Precision for a set of line-of-sight vectors.

    GDOP = sqrt(trace((H^T H)^-1)) where each row of H is a unit LOS vector.
    Lower is better (observations more spread out / better conditioned).
    """
    h = np.array(los_vectors)
    try:
        cov = np.linalg.inv(h.T @ h)
    except np.linalg.LinAlgError:
        return np.inf
    trace = np.trace(cov)
    return np.sqrt(trace) if trace > 0 else np.inf


def select_optimal_triplet(los_vectors: list[np.ndarray], indices: list | None = None,
                           min_sep_deg: float = 3.0,
                           max_sep_deg: float = 60.0) -> dict | None:
    """Pick the best 3-observation subset from many.

    Preference order:
      1. first-to-last angular separation inside ``[min_sep_deg, max_sep_deg]``
         (avoids the near-coplanar singularity while keeping the arc short
         enough for the series expansions to hold);
      2. lowest GDOP among the qualifying triplets.

    Returns a dict with the chosen indices, GDOP, separation and a flag saying
    whether the ideal separation window was met, or ``None`` if fewer than three
    observations are supplied.
    """
    n = len(los_vectors)
    if n < 3:
        return None
    if indices is None:
        indices = list(range(n))

    ideal, fallback = [], []
    for combo in combinations(range(n), 3):
        i, j, k = combo
        sep = angular_separation_deg(los_vectors[i], los_vectors[k])
        gdop = compute_gdop([los_vectors[i], los_vectors[j], los_vectors[k]])
        entry = (gdop, sep, combo)
        fallback.append(entry)
        if min_sep_deg <= sep <= max_sep_deg:
            ideal.append(entry)

    pool = ideal if ideal else fallback
    pool.sort(key=lambda c: c[0])
    best = pool[0]
    return {
        "indices": [indices[x] for x in best[2]],
        "gdop": best[0],
        "angular_sep_deg": best[1],
        "in_ideal_range": bool(ideal),
    }
