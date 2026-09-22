"""
================================================================================
 db.py -- Central results database (SQLite)
================================================================================
A single self-contained database that stores every object, station, observation
and IOD solution with the fields required by the specification:

  Object ID, Observation ID, Time, Station ID, Observation Type,
  Measurement Values, Units, Reference Frame, IOD Method, Input Dataset,
  State Vector, Orbital Elements, Validation Results, Processing Status,
  Software Version.

SQLite keeps the whole catalogue in one portable file with zero setup, while
still supporting relational queries, indices and joins across the four tables.
================================================================================
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .constants import SOFTWARE_VERSION


_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    object_id      TEXT PRIMARY KEY,
    label          TEXT,
    created_utc    TEXT
);

CREATE TABLE IF NOT EXISTS stations (
    station_id     TEXT PRIMARY KEY,
    name           TEXT,
    latitude_deg   REAL,
    longitude_deg  REAL,
    altitude_km    REAL,
    ecef_x_km      REAL,
    ecef_y_km      REAL,
    ecef_z_km      REAL
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id          TEXT,
    epoch_utc          TEXT,
    station_id         TEXT,
    observation_type   TEXT,
    measurement_values TEXT,      -- JSON list
    units              TEXT,      -- JSON dict
    reference_frame    TEXT,
    input_dataset      TEXT,
    FOREIGN KEY(object_id)  REFERENCES objects(object_id),
    FOREIGN KEY(station_id) REFERENCES stations(station_id)
);

CREATE TABLE IF NOT EXISTS iod_solutions (
    solution_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id          TEXT,
    iod_method         TEXT,
    input_dataset      TEXT,
    epoch_utc          TEXT,
    reference_frame    TEXT,
    state_vector       TEXT,      -- JSON {x,y,z,vx,vy,vz}
    orbital_elements   TEXT,      -- JSON full element set
    validation_results TEXT,      -- JSON
    processing_status  TEXT,
    software_version   TEXT,
    created_utc        TEXT,
    FOREIGN KEY(object_id) REFERENCES objects(object_id)
);

CREATE INDEX IF NOT EXISTS idx_obs_object  ON observations(object_id);
CREATE INDEX IF NOT EXISTS idx_sol_object  ON iod_solutions(object_id);
CREATE INDEX IF NOT EXISTS idx_sol_method  ON iod_solutions(iod_method);
"""


class IODDatabase:
    def __init__(self, path: str = "iod_results.db"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- inserts ----------------------------------------------------------
    def upsert_object(self, object_id: str, label: str = ""):
        self.conn.execute(
            "INSERT OR IGNORE INTO objects(object_id, label, created_utc) "
            "VALUES (?,?,?)",
            (object_id, label, datetime.utcnow().isoformat()))
        self.conn.commit()

    def upsert_station(self, station_id, name, lat, lon, alt, ecef=None):
        ex, ey, ez = (ecef if ecef is not None else (None, None, None))
        self.conn.execute(
            "INSERT OR REPLACE INTO stations(station_id, name, latitude_deg, "
            "longitude_deg, altitude_km, ecef_x_km, ecef_y_km, ecef_z_km) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (station_id, name, lat, lon, alt, ex, ey, ez))
        self.conn.commit()

    def add_observation(self, object_id, epoch_utc, station_id, obs_type,
                        measurement_values, units, reference_frame="GCRF",
                        input_dataset="") -> int:
        cur = self.conn.execute(
            "INSERT INTO observations(object_id, epoch_utc, station_id, "
            "observation_type, measurement_values, units, reference_frame, "
            "input_dataset) VALUES (?,?,?,?,?,?,?,?)",
            (object_id, epoch_utc, station_id, obs_type,
             json.dumps(measurement_values), json.dumps(units),
             reference_frame, input_dataset))
        self.conn.commit()
        return cur.lastrowid

    def add_solution(self, object_id, iod_method, epoch_utc, state_vector,
                     orbital_elements, validation_results, processing_status,
                     reference_frame="GCRF", input_dataset="") -> int:
        cur = self.conn.execute(
            "INSERT INTO iod_solutions(object_id, iod_method, input_dataset, "
            "epoch_utc, reference_frame, state_vector, orbital_elements, "
            "validation_results, processing_status, software_version, "
            "created_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (object_id, iod_method, input_dataset, epoch_utc, reference_frame,
             json.dumps(state_vector), json.dumps(orbital_elements),
             json.dumps(validation_results), processing_status,
             SOFTWARE_VERSION, datetime.utcnow().isoformat()))
        self.conn.commit()
        return cur.lastrowid

    # -- queries ----------------------------------------------------------
    def solutions_for(self, object_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM iod_solutions WHERE object_id=? ORDER BY solution_id",
            (object_id,)).fetchall()
        return [self._row_to_solution(r) for r in rows]

    def all_solutions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM iod_solutions ORDER BY solution_id").fetchall()
        return [self._row_to_solution(r) for r in rows]

    @staticmethod
    def _row_to_solution(r: sqlite3.Row) -> dict:
        return {
            "solution_id": r["solution_id"],
            "object_id": r["object_id"],
            "iod_method": r["iod_method"],
            "epoch_utc": r["epoch_utc"],
            "reference_frame": r["reference_frame"],
            "state_vector": json.loads(r["state_vector"]),
            "orbital_elements": json.loads(r["orbital_elements"]),
            "validation_results": json.loads(r["validation_results"]),
            "processing_status": r["processing_status"],
            "software_version": r["software_version"],
        }

    def close(self):
        self.conn.close()
