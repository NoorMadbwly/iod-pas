"""
IOD-PAS -- Initial Orbit Determination Processing & Analysis System
===================================================================
A unified engine implementing all six required IOD methods behind one interface:

    Angles-only : Gauss, Laplace
    Range       : Range + Range-Rate, Range-Only
    Position    : Gibbs, Herrick-Gibbs

plus smart ingestion, a ground-station catalogue, full orbital-element output,
physical validation, and a central results database.
"""

from .constants import MU_EARTH, R_EARTH_EQ, SOFTWARE_VERSION
from .elements import rv_to_elements, elements_to_rv
from .propagate import propagate_two_body
from .gauss import gauss_method, GaussRootAmbiguityError
from .laplace import laplace_method
from .gibbs import (gibbs_velocity, herrick_gibbs_velocity,
                    compute_velocity_auto)
from .range_iod import range_iod
from .validation import sanity_check_result, UnphysicalResultError
from .ingest import load_observation_file
from .stations import StationCatalogue, Station
from .db import IODDatabase
from .pipeline import (IODResult, solve_gauss, solve_laplace, solve_gibbs,
                       solve_range)

__version__ = SOFTWARE_VERSION
