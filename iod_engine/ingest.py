"""
================================================================================
 ingest.py -- Smart ingestion and preprocessing layer
================================================================================
Reads a raw observation file, auto-detects its format from the header, and emits
a single normalised dataset regardless of source. Handles:
  * automatic format detection (state-vector "ASN", range/range-rate "ITNP");
  * continuous time reconstruction (date + seconds, rolling past midnight);
  * exact-duplicate removal;
  * unit sanity checks (metres -> km, m/s -> km/s) with a transparency report.

The canonical observation containers below are also what the angle and range
methods consume, so a new sensor format only needs a new reader here.
================================================================================
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# --------------------------------------------------------------------------
# Canonical containers
# --------------------------------------------------------------------------

@dataclass
class PreprocessReport:
    file_path: str
    detected_format: str
    warnings: list = field(default_factory=list)
    fixes_applied: list = field(default_factory=list)
    n_raw_rows: int = 0
    n_duplicates_removed: int = 0
    n_final_rows: int = 0

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_fix(self, msg: str):
        self.fixes_applied.append(msg)

    def summary(self) -> str:
        lines = [f"[Preprocessing report] {self.file_path}",
                 f"  detected format : {self.detected_format}",
                 f"  raw rows        : {self.n_raw_rows}",
                 f"  duplicates cut  : {self.n_duplicates_removed}",
                 f"  final rows      : {self.n_final_rows}"]
        if self.fixes_applied:
            lines.append("  automatic fixes:")
            lines += [f"    - {f}" for f in self.fixes_applied]
        if self.warnings:
            lines.append("  warnings (need human review):")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class StateVectorObservation:
    epoch: datetime
    r_eci_km: np.ndarray
    v_eci_kms: np.ndarray


@dataclass
class RangeRateObservation:
    epoch: datetime
    range_km: float
    range_rate_kms: float


@dataclass
class AngleObservation:
    epoch: datetime
    angle1_deg: float          # azimuth / right ascension / hour angle
    angle2_deg: float          # elevation / declination
    angle_type: str            # "azel" | "radec" | "hadec"
    range_km: Optional[float] = None


@dataclass
class NormalizedDataset:
    header_meta: dict
    kind: str                  # "state_vector" | "range_rate" | "angles"
    observations: list
    report: PreprocessReport


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_HEADER_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _parse_header_kv(line: str) -> dict:
    return dict(_HEADER_KV_RE.findall(line))


def _parse_date_ddmmyyyy(s: str) -> datetime:
    d, m, y = (int(x) for x in s.split("."))
    if m > 12 and d <= 12:      # columns swapped -> MM.DD.YYYY
        d, m = m, d
    return datetime(y, m, d)


def _seconds_to_epoch(base_date: datetime, seconds: float) -> datetime:
    days, rem = divmod(seconds, 86400.0)
    return base_date + timedelta(days=days, seconds=rem)


def _dedup(rows):
    out = []
    for row in rows:
        if out and np.allclose(out[-1], row, atol=1e-6):
            continue
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Format readers
# --------------------------------------------------------------------------

def _read_asn(lines, path) -> NormalizedDataset:
    meta = _parse_header_kv(lines[0])
    report = PreprocessReport(file_path=path, detected_format="ASN (state vector)")
    base_date = _parse_date_ddmmyyyy(meta["DAT"]) if "DAT" in meta else None
    if base_date is None:
        report.add_warning("no date (DAT) in header; assumed 2000-01-01, verify.")
        base_date = datetime(2000, 1, 1)

    raw = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 7:
            report.add_warning(f"row with {len(parts)} values (expected 7) skipped.")
            continue
        raw.append([float(x) for x in parts])

    report.n_raw_rows = len(raw)
    dedup = _dedup(raw)
    report.n_duplicates_removed = report.n_raw_rows - len(dedup)
    if report.n_duplicates_removed:
        report.add_fix(f"removed {report.n_duplicates_removed} exact duplicate rows.")

    arr = np.array(dedup)
    pos_scale = 1.0
    if len(arr) and np.linalg.norm(arr[:, 1:4], axis=1).mean() > 1e5:
        pos_scale = 1e-3
        report.add_fix("positions looked like metres; converted to km.")
    vel_scale = 1.0
    if len(arr) and np.linalg.norm(arr[:, 4:7], axis=1).mean() > 50:
        vel_scale = 1e-3
        report.add_fix("velocities looked like m/s; converted to km/s.")

    obs = []
    for row in dedup:
        t_sec, x, y, z, vx, vy, vz = row
        obs.append(StateVectorObservation(
            epoch=_seconds_to_epoch(base_date, t_sec),
            r_eci_km=np.array([x, y, z]) * pos_scale,
            v_eci_kms=np.array([vx, vy, vz]) * vel_scale))

    if any(obs[i].epoch > obs[i + 1].epoch for i in range(len(obs) - 1)):
        obs.sort(key=lambda o: o.epoch)
        report.add_fix("rows were not time-ordered; sorted ascending.")

    report.n_final_rows = len(obs)
    return NormalizedDataset(meta, "state_vector", obs, report)


def _read_itnp(lines, path) -> NormalizedDataset:
    meta = _parse_header_kv(lines[0])
    report = PreprocessReport(file_path=path,
                              detected_format="ITNP (range/range-rate)")
    base_date = (_parse_date_ddmmyyyy(meta["DAT"]) if "DAT" in meta
                 else datetime(2000, 1, 1))

    raw = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 3:
            report.add_warning(f"row with {len(parts)} values (expected 3) skipped.")
            continue
        raw.append([float(x) for x in parts])

    report.n_raw_rows = len(raw)
    dedup = _dedup(raw)
    report.n_duplicates_removed = report.n_raw_rows - len(dedup)

    arr = np.array(dedup)
    scale = 1e-3 if len(arr) and np.abs(arr[:, 1]).mean() > 1e5 else 1.0
    if scale != 1.0:
        report.add_fix("range looked like metres; converted to km.")

    obs = [RangeRateObservation(
        epoch=_seconds_to_epoch(base_date, row[0]),
        range_km=row[1] * scale, range_rate_kms=row[2] * scale)
        for row in dedup]

    report.n_final_rows = len(obs)
    return NormalizedDataset(meta, "range_rate", obs, report)


def load_observation_file(path: str) -> NormalizedDataset:
    """Read any supported observation file and return a normalised dataset."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\r\n") for ln in f.readlines()]

    if len(lines) < 2:
        raise ValueError(f"{path} is empty or missing a header.")

    tag = lines[1].strip().upper()
    if tag == "ASN":
        return _read_asn(lines, path)
    if tag == "ITNP":
        return _read_itnp(lines, path)
    raise ValueError(
        f"Unknown format in {path} (header tag {tag!r}); add a reader for it.")
