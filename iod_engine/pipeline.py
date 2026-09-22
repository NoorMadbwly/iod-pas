"""
================================================================================
 pipeline.py -- Unified input/output orchestration layer
================================================================================
One entry point that turns "a dataset + a chosen method" into a fully populated,
validated, storable solution, no matter which of the six methods is requested.
This realises the specification's "Unified Input/Output Concept": every method
returns the same result object (epoch, state vector, full orbital elements,
validation block, processing status), so the database and dashboard never need
to know which method produced a given solution.
================================================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .constants import SOFTWARE_VERSION
from .elements import rv_to_elements
from .validation import sanity_check_result, UnphysicalResultError
from .gauss import gauss_method
from .laplace import laplace_method
from .gibbs import compute_velocity_auto
from .range_iod import range_iod


@dataclass
class IODResult:
    method: str
    object_id: str
    epoch: Optional[datetime]
    r_eci_km: np.ndarray
    v_eci_kms: np.ndarray
    elements: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    status: str = "PENDING"
    reference_frame: str = "GCRF"
    input_dataset: str = ""
    extra: dict = field(default_factory=dict)

    def state_vector_dict(self) -> dict:
        return {"x": float(self.r_eci_km[0]), "y": float(self.r_eci_km[1]),
                "z": float(self.r_eci_km[2]), "vx": float(self.v_eci_kms[0]),
                "vy": float(self.v_eci_kms[1]), "vz": float(self.v_eci_kms[2])}

    def store(self, db) -> int:
        db.upsert_object(self.object_id)
        return db.add_solution(
            object_id=self.object_id, iod_method=self.method,
            epoch_utc=self.epoch.isoformat() if self.epoch else "",
            state_vector=self.state_vector_dict(),
            orbital_elements=self.elements,
            validation_results=self.validation,
            processing_status=self.status,
            reference_frame=self.reference_frame,
            input_dataset=self.input_dataset)


def _finalize(method, object_id, r2, v2, epoch, input_dataset="",
              extra=None) -> IODResult:
    """Common tail: compute elements, run the physical gate, set status."""
    extra = extra or {}
    result = IODResult(method=method, object_id=object_id, epoch=epoch,
                       r_eci_km=np.asarray(r2, float),
                       v_eci_kms=np.asarray(v2, float),
                       input_dataset=input_dataset, extra=extra)
    try:
        elements = sanity_check_result(result.r_eci_km, result.v_eci_kms)
        result.elements = elements
        result.validation = {"physical_gate": "PASS", "software_version": SOFTWARE_VERSION}
        result.status = "VALIDATED"
    except UnphysicalResultError as exc:
        result.elements = rv_to_elements(result.r_eci_km, result.v_eci_kms)
        result.validation = {"physical_gate": "FAIL", "reason": str(exc),
                             "software_version": SOFTWARE_VERSION}
        result.status = "REJECTED"
    if extra:
        result.validation.update({k: v for k, v in extra.items()
                                  if np.isscalar(v)})
    return result


# --------------------------------------------------------------------------
# Method-specific solvers (all return an IODResult)
# --------------------------------------------------------------------------

def solve_gauss(object_id, los, site_eci, times_sec, epoch, **kw) -> IODResult:
    out = gauss_method(los, site_eci, times_sec, **kw)
    res = _finalize("Gauss", object_id, out["r2"], out["v2"], epoch,
                    extra={"gauss_iterations": out["diagnostics"].get("iterations")})
    res.extra["raw"] = out
    return res


def solve_laplace(object_id, los, site_states, times_sec, epoch) -> IODResult:
    out = laplace_method(los, site_states, times_sec)
    res = _finalize("Laplace", object_id, out["r2"], out["v2"], epoch)
    res.extra["raw"] = out
    return res


def solve_gibbs(object_id, r1, r2, r3, times_sec, epoch) -> IODResult:
    t1, t2, t3 = times_sec
    vel = compute_velocity_auto(r1, r2, r3, t1, t2, t3)
    res = _finalize(vel["method_used"].split(" ")[0], object_id, r2, vel["v2"],
                    epoch, extra={"velocity_method": vel["method_used"],
                                  "coplanarity_deviation_deg":
                                      vel["coplanarity_deviation_deg"]})
    res.method = vel["method_used"]
    res.extra["raw"] = vel
    return res


def solve_range(object_id, observations, epoch, use_range_rate=True,
                initial_guess=None) -> IODResult:
    out = range_iod(observations, initial_guess=initial_guess,
                    use_range_rate=use_range_rate)
    method = "Range+Range-Rate" if use_range_rate else "Range-Only"
    res = _finalize(method, object_id, out["r0"], out["v0"], epoch,
                    extra={"rms_residual": out["rms_residual"],
                           "converged": out["converged"],
                           "iterations": out["iterations"]})
    res.extra["raw"] = out
    return res
