"""Public interface for PDECert."""

from .benchmark import BENCHMARK_VERSION, BenchmarkError, evaluate_corpus
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
from .corpus import (
    ANNOTATION_STATUSES,
    CORPUS_VERSION,
    FAILURE_MODES,
    ORIGIN_KINDS,
    VERDICTS,
    CorpusError,
    dump_corpus,
    load_corpus,
    output_sha256,
    validate_corpus,
)
from .labeling import REVIEW_VERSION, ReviewError, apply_review
from .schema import (
    SchemaError,
    VerificationCase,
    case_from_dict,
    case_to_dict,
    dump_case,
    load_case,
)

__all__ = [
    "ANNOTATION_STATUSES",
    "BENCHMARK_VERSION",
    "BenchmarkError",
    "CORPUS_VERSION",
    "Constraint",
    "CorpusError",
    "FAILURE_MODES",
    "ORIGIN_KINDS",
    "PARAMETER_ASSUMPTIONS",
    "Problem",
    "REVIEW_VERSION",
    "Report",
    "ReviewError",
    "SchemaError",
    "Status",
    "VerificationCase",
    "VERDICTS",
    "Witness",
    "case_from_dict",
    "case_to_dict",
    "apply_review",
    "dump_case",
    "dump_corpus",
    "evaluate_corpus",
    "fixed_collocation_check",
    "load_case",
    "load_corpus",
    "output_sha256",
    "validate_corpus",
    "verify",
]

__version__ = "0.1.0"
