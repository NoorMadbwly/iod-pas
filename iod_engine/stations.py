"""
================================================================================
 stations.py -- Ground-station catalogue
================================================================================
Loads the supplied station workbook and exposes a simple lookup. The workbook
carries two sheets with different layouts:

  Sheet 1 : name, geodetic latitude/longitude/altitude, and Earth-fixed X,Y,Z.
  Sheet 2 : name, Earth-fixed X,Y,Z only (this sheet contains the Egyptian
            stations "cairo" and "Aswan").

Both are merged into one catalogue keyed by (lower-cased) name. Where only ECEF
is available, geodetic coordinates are recovered from the ellipsoid so every
station can be placed in ECI at an arbitrary epoch.
================================================================================
"""

from __future__ import annotations

import os
import numpy as np
from datetime import datetime

from .frames import ecef_to_geodetic, station_geodetic_to_eci, station_state_eci

try:                                    # openpyxl-backed loading
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:                       # pragma: no cover
    _HAVE_PANDAS = False

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                             "ground_stations.xlsx")


class Station:
    def __init__(self, name, lat, lon, alt, ecef=None):
        self.name = name
        self.lat_deg = lat
        self.lon_deg = lon
        self.alt_km = alt
        self.ecef_km = ecef

    def eci_position(self, epoch: datetime) -> np.ndarray:
        return station_geodetic_to_eci(self.lat_deg, self.lon_deg,
                                       self.alt_km, epoch)

    def eci_state(self, epoch: datetime):
        return station_state_eci(self.lat_deg, self.lon_deg, self.alt_km, epoch)

    def __repr__(self):
        return (f"Station({self.name!r}, lat={self.lat_deg:.4f}, "
                f"lon={self.lon_deg:.4f}, alt={self.alt_km:.3f} km)")


class StationCatalogue:
    def __init__(self, path: str | None = None):
        self._stations: dict[str, Station] = {}
        self.path = path or _DEFAULT_PATH
        if _HAVE_PANDAS and os.path.exists(self.path):
            self._load()

    def _load(self):
        xl = pd.ExcelFile(self.path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            cols = {str(c).strip().lower(): c for c in df.columns}
            name_col = next((cols[c] for c in cols
                             if "station" in c or "name" in c or "neam" in c),
                            None)
            if name_col is None:
                continue
            has_geo = "lat" in cols and any(c.startswith("long") for c in cols)
            lon_col = next((cols[c] for c in cols if c.startswith("long")), None)
            alt_col = next((cols[c] for c in cols
                            if c in ("h/m", "alt", "altitude", "h")), None)
            x_col = cols.get("x")
            y_col = cols.get("y")
            z_col = cols.get("z")

            for _, row in df.iterrows():
                name = row[name_col]
                if not isinstance(name, str) or not name.strip():
                    continue
                key = name.strip().lower()
                if key in self._stations:
                    continue
                if has_geo and not _isnan(row[cols["lat"]]):
                    lat = float(row[cols["lat"]])
                    lon = float(row[lon_col])
                    alt = float(row[alt_col]) / 1000.0 if alt_col else 0.0
                    ecef = (np.array([row[x_col], row[y_col], row[z_col]],
                                     dtype=float)
                            if x_col and not _isnan(row[x_col]) else None)
                    self._stations[key] = Station(name.strip(), lat, lon, alt, ecef)
                elif x_col and not _isnan(row[x_col]):
                    ecef = np.array([row[x_col], row[y_col], row[z_col]],
                                    dtype=float)
                    lat, lon, alt = ecef_to_geodetic(ecef)
                    self._stations[key] = Station(name.strip(), lat, lon, alt, ecef)

    def get(self, name: str) -> Station | None:
        return self._stations.get(name.strip().lower())

    def names(self):
        return sorted(s.name for s in self._stations.values())

    def __len__(self):
        return len(self._stations)


def _isnan(x) -> bool:
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return True
