"""Public interface for PDECert."""

from .core import (
    PARAMETER_ASSUMPTIONS,
    Constraint,
    Problem,
    Report,
    Status,
    Witness,
    fixed_collocation_check,
    verify,
)
from .schema import (
    SchemaError,
    VerificationCase,
    case_from_dict,
    case_to_dict,
    dump_case,
    load_case,
)

__all__ = [
    "Constraint",
    "PARAMETER_ASSUMPTIONS",
    "Problem",
    "Report",
    "SchemaError",
    "Status",
    "VerificationCase",
    "Witness",
    "case_from_dict",
    "case_to_dict",
    "dump_case",
    "fixed_collocation_check",
    "load_case",
    "verify",
]

__version__ = "0.1.0"
