"""
================================================================================
 constants.py -- Shared physical and reference constants (IOD-PAS)
================================================================================
All values in SI-derived kilometre / second units unless stated otherwise.
Sources: IAU / IERS conventions, WGS-84 reference ellipsoid.
================================================================================
"""

from __future__ import annotations

# Gravitational parameter of Earth (km^3 / s^2), EGM-96 / WGS-84 value.
MU_EARTH = 398600.4418

# WGS-84 reference ellipsoid.
R_EARTH_EQ = 6378.137            # equatorial radius (km)
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_E2 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)   # first eccentricity squared

# Mean equatorial radius kept for quick spherical checks.
R_EARTH = R_EARTH_EQ

# Earth inertial rotation rate (rad/s), IERS value.
OMEGA_EARTH = 7.292115146706979e-5

# Standard second-per-degree conversion for sidereal time bookkeeping.
SECONDS_PER_DAY = 86400.0

# Project software version tag stored with every solution.
SOFTWARE_VERSION = "IOD-PAS v1.0"
