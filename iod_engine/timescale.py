"""
================================================================================
 timescale.py -- Julian date and sidereal-time utilities
================================================================================
Converts civil UTC epochs to Julian date and Greenwich / local sidereal time.
The GMST model is the standard IAU-1982 polynomial, accurate to well below the
level required for initial orbit determination.
================================================================================
"""

from __future__ import annotations

import numpy as np
from datetime import datetime

from .constants import SECONDS_PER_DAY


def julian_date(dt: datetime) -> float:
    """Convert a (UTC) datetime to Julian Date."""
    y, m = dt.year, dt.month
    day = dt.day + (dt.hour + dt.minute / 60.0
                    + dt.second / 3600.0
                    + dt.microsecond / 3.6e9) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1))
            + day + b - 1524.5)


def gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU-1982 polynomial)."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * t
                + 0.093104 * t ** 2
                - 6.2e-6 * t ** 3)
    gmst_deg = (gmst_sec % SECONDS_PER_DAY) / 240.0   # 240 s == 1 deg
    return np.radians(gmst_deg % 360.0)


def local_sidereal_time_rad(dt: datetime, lon_deg: float) -> float:
    """Local mean sidereal time (radians) at east longitude ``lon_deg``."""
    return (gmst_rad(dt) + np.radians(lon_deg)) % (2.0 * np.pi)
