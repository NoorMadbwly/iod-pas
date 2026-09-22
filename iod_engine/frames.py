"""
================================================================================
 frames.py -- Coordinate frames, ground-station kinematics, angle adapters
================================================================================
Provides:
  * geodetic  <-> ECEF        (WGS-84 ellipsoid, not a sphere)
  * ECEF      <-> ECI         (sidereal rotation about the pole)
  * ground-station position / velocity / acceleration in ECI
  * a unified "angle adapter" turning any supported angle pair
    (Az/El, RA/Dec, Hour-Angle/Dec) into an ECI line-of-sight unit vector.

The angle-adapter design lets the three IOD engineers share a single
observation front-end regardless of the coordinate system a station reports in.
================================================================================
"""

from __future__ import annotations

import numpy as np
from datetime import datetime

from .constants import R_EARTH_EQ, WGS84_E2, OMEGA_EARTH
from .timescale import gmst_rad, local_sidereal_time_rad


# --------------------------------------------------------------------------
# Geodetic <-> ECEF (WGS-84 ellipsoid)
# --------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """Geodetic latitude/longitude/altitude -> Earth-fixed (ECEF) position (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    n = R_EARTH_EQ / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (n + alt_km) * np.cos(lat) * np.cos(lon)
    y = (n + alt_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_km) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """Earth-fixed position (km) -> geodetic (lat_deg, lon_deg, alt_km).

    Bowring's iterative method; converges in a couple of iterations.
    """
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        sin_lat = np.sin(lat)
        n = R_EARTH_EQ / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return np.degrees(lat), np.degrees(lon), alt


# --------------------------------------------------------------------------
# ECEF <-> ECI  (rotation about the pole by sidereal angle)
# --------------------------------------------------------------------------

def _rot3(theta: float) -> np.ndarray:
    """Rotation of a coordinate frame by ``theta`` about the z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def ecef_to_eci(r_ecef: np.ndarray, epoch: datetime) -> np.ndarray:
    """Earth-fixed -> inertial position at ``epoch`` (GMST rotation)."""
    theta = gmst_rad(epoch)
    return _rot3(theta).T @ r_ecef        # inverse of the ECI->ECEF rotation


def eci_to_ecef(r_eci: np.ndarray, epoch: datetime) -> np.ndarray:
    """Inertial -> Earth-fixed position at ``epoch``."""
    theta = gmst_rad(epoch)
    return _rot3(theta) @ r_eci


# --------------------------------------------------------------------------
# Ground-station kinematics in ECI
# --------------------------------------------------------------------------

def station_geodetic_to_eci(lat_deg: float, lon_deg: float, alt_km: float,
                            epoch: datetime) -> np.ndarray:
    """Ground-station geodetic coordinates -> ECI position (km) at ``epoch``."""
    return ecef_to_eci(geodetic_to_ecef(lat_deg, lon_deg, alt_km), epoch)


def station_state_eci(lat_deg: float, lon_deg: float, alt_km: float,
                      epoch: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return station (position, velocity, acceleration) in ECI at ``epoch``.

    Velocity and acceleration arise purely from Earth rotation:
        v = omega x r,   a = omega x (omega x r).
    These are required by the Laplace method.
    """
    r = station_geodetic_to_eci(lat_deg, lon_deg, alt_km, epoch)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v = np.cross(omega, r)
    a = np.cross(omega, v)
    return r, v, a


def station_ecef_to_eci(r_ecef: np.ndarray, epoch: datetime):
    """Station given directly as ECEF (km) -> (position, velocity, accel) in ECI."""
    r = ecef_to_eci(r_ecef, epoch)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v = np.cross(omega, r)
    a = np.cross(omega, v)
    return r, v, a


# --------------------------------------------------------------------------
# Angle adapter -> ECI line-of-sight unit vector
# --------------------------------------------------------------------------

def azel_to_los_eci(az_deg: float, el_deg: float, lat_deg: float,
                    lon_deg: float, epoch: datetime) -> np.ndarray:
    """Azimuth/elevation (topocentric horizon, System 1) -> ECI unit vector.

    Azimuth measured from North, positive toward East; elevation above the
    local horizon. Requires the full site rotation (latitude + LST).
    """
    az, el = np.radians(az_deg), np.radians(el_deg)
    # South-East-Zenith (SEZ) topocentric components.
    rho_sez = np.array([-np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el)])
    lat = np.radians(lat_deg)
    lst = local_sidereal_time_rad(epoch, lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    st, ct = np.sin(lst), np.cos(lst)
    rot = np.array([[sl * ct, -st, cl * ct],
                    [sl * st, ct, cl * st],
                    [-cl, 0.0, sl]])
    los = rot @ rho_sez
    return los / np.linalg.norm(los)


def radec_to_los_eci(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Right-ascension/declination (topocentric equatorial, System 2) -> ECI unit vector.

    The equatorial axes are already parallel to ECI, so no sidereal rotation
    is needed -- this is the classical form used by Gauss and Laplace.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def hadec_to_los_eci(ha_deg: float, dec_deg: float, lon_deg: float,
                     epoch: datetime) -> np.ndarray:
    """Hour-angle/declination (System 4) -> ECI unit vector.

    Right ascension is recovered from RA = LST - HA, then reduced to the
    equatorial line-of-sight. Useful for stations that report hour angle.
    """
    lst_deg = np.degrees(local_sidereal_time_rad(epoch, lon_deg))
    ra_deg = (lst_deg - ha_deg) % 360.0
    return radec_to_los_eci(ra_deg, dec_deg)


def angle_observation_to_los_eci(obs, station_lat_deg: float,
                                 station_lon_deg: float) -> np.ndarray:
    """Dispatch an :class:`AngleObservation` to the correct adapter."""
    kind = obs.angle_type
    if kind == "azel":
        return azel_to_los_eci(obs.angle1_deg, obs.angle2_deg,
                               station_lat_deg, station_lon_deg, obs.epoch)
    if kind == "radec":
        return radec_to_los_eci(obs.angle1_deg, obs.angle2_deg)
    if kind == "hadec":
        return hadec_to_los_eci(obs.angle1_deg, obs.angle2_deg,
                                station_lon_deg, obs.epoch)
    raise ValueError(f"Unsupported angle type: {kind!r}")
