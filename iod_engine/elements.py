"""
================================================================================
 elements.py -- Classical orbital elements <-> state vector
================================================================================
Full six-element set required by the project specification:
    a   -- semi-major axis            (km)
    e   -- eccentricity               (-)
    i   -- inclination                (deg)
    RAAN(Omega) -- right ascension of the ascending node (deg)
    argp(omega) -- argument of perigee                    (deg)
    nu  -- true anomaly               (deg)
Plus derived quantities (mean anomaly, period, apogee/perigee altitude).

Robust against the classical singular cases (circular and/or equatorial orbits).
================================================================================
"""

from __future__ import annotations

import numpy as np

from .constants import MU_EARTH, R_EARTH_EQ

_TOL = 1e-9


def rv_to_elements(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> dict:
    """State vector (r, v) in ECI -> classical orbital elements.

    Handles circular and equatorial degeneracies gracefully by falling back to
    the appropriate special-case anomaly definitions.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector (points toward the ascending node).
    k_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector.
    e_vec = (np.cross(v, h) / mu) - (r / r_mag)
    e = np.linalg.norm(e_vec)

    # Specific energy -> semi-major axis.
    energy = v_mag ** 2 / 2.0 - mu / r_mag
    if abs(energy) > _TOL:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf   # parabolic

    inc = np.degrees(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0)))

    # Right ascension of the ascending node.
    if n_mag > _TOL:
        raan = np.degrees(np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0)))
        if n[1] < 0:
            raan = 360.0 - raan
    else:
        raan = 0.0   # equatorial: node undefined, set to zero by convention

    # Argument of perigee.
    if n_mag > _TOL and e > _TOL:
        argp = np.degrees(np.arccos(
            np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 360.0 - argp
    else:
        argp = 0.0

    # True anomaly (with special cases).
    if e > _TOL:
        nu = np.degrees(np.arccos(
            np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0)))
        if np.dot(r, v) < 0:
            nu = 360.0 - nu
    elif n_mag > _TOL:
        # Circular inclined: use argument of latitude.
        nu = np.degrees(np.arccos(
            np.clip(np.dot(n, r) / (n_mag * r_mag), -1.0, 1.0)))
        if r[2] < 0:
            nu = 360.0 - nu
    else:
        # Circular equatorial: use true longitude.
        nu = np.degrees(np.arccos(np.clip(r[0] / r_mag, -1.0, 1.0)))
        if r[1] < 0:
            nu = 360.0 - nu

    # Mean anomaly from eccentric anomaly (elliptical only).
    mean_anomaly = None
    if 0 <= e < 1:
        nu_rad = np.radians(nu)
        ecc_anom = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(nu_rad),
                              e + np.cos(nu_rad))
        mean_anomaly = np.degrees((ecc_anom - e * np.sin(ecc_anom)) % (2 * np.pi))

    period = (2.0 * np.pi * np.sqrt(a ** 3 / mu)
              if np.isfinite(a) and a > 0 else None)

    apogee_alt = a * (1 + e) - R_EARTH_EQ if np.isfinite(a) else None
    perigee_alt = a * (1 - e) - R_EARTH_EQ if np.isfinite(a) else None

    return {
        "semi_major_axis_km": float(a),
        "eccentricity": float(e),
        "inclination_deg": float(inc),
        "raan_deg": float(raan),
        "arg_perigee_deg": float(argp),
        "true_anomaly_deg": float(nu),
        "mean_anomaly_deg": (float(mean_anomaly)
                             if mean_anomaly is not None else None),
        "specific_energy": float(energy),
        "angular_momentum_km2_s": float(h_mag),
        "period_s": float(period) if period is not None else None,
        "apogee_altitude_km": (float(apogee_alt)
                               if apogee_alt is not None else None),
        "perigee_altitude_km": (float(perigee_alt)
                                if perigee_alt is not None else None),
    }


def elements_to_rv(a: float, e: float, inc_deg: float, raan_deg: float,
                   argp_deg: float, nu_deg: float, mu: float = MU_EARTH):
    """Classical elements -> state vector (r, v) in ECI.

    Used by the visualisation layer and for round-trip verification of
    :func:`rv_to_elements`.
    """
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(nu_deg)

    p = a * (1 - e ** 2)                      # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))

    # Position and velocity in the perifocal frame.
    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cr, sr = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    # Perifocal -> ECI rotation (3-1-3 sequence).
    rot = np.array([
        [cr * cw - sr * sw * ci, -cr * sw - sr * cw * ci, sr * si],
        [sr * cw + cr * sw * ci, -sr * sw + cr * cw * ci, -cr * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pqw, rot @ v_pqw
