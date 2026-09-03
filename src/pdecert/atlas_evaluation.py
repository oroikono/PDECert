"""Evidence-preserving evaluation of typed Atlas v2 records."""

from __future__ import annotations

import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

import sympy

from .artifacts import SymbolicCandidate
from .compiler import OperatorCompileError, compile_autodiff_problem
from .core import verify, verify_artifact
from .corpus import (
    CROSS_ARTIFACT_ATLAS_VERSION,
    CorpusError,
    cross_artifact_atlas_sha256,
    load_cross_artifact_atlas,
)
from .frozen_callable import (
    FrozenCallableError,
    frozen_callable_from_dict,
    materialize_frozen_callable,
)
from .templates import TemplateError, bind_symbolic_candidate, template_from_dict


ATLAS_EVALUATION_VERSION = 1


class AtlasEvaluationError(ValueError):
    """Raised when a typed Atlas record cannot be evaluated under this contract."""


@dataclass(frozen=True)
class AtlasEvaluationOptions:
    """Reproducible options for symbolic and callable Atlas evaluators."""

    symbolic_tolerance: float = 1e-9
    callable_tolerance: float = 1e-6
    samples_per_axis: int = 5
    symbolic_timeout: float = 2.0
    max_expression_ops: int = 10_000

    def __post_init__(self) -> None:
        for name in ("symbolic_tolerance", "callable_tolerance", "symbolic_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("samples_per_axis", 2), ("max_expression_ops", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")

    def to_dict(self) -> dict[str, object]:
        """Return strict-JSON-compatible evaluator settings."""

        return {
            "callable_tolerance": self.callable_tolerance,
            "max_expression_ops": self.max_expression_ops,
            "samples_per_axis": self.samples_per_axis,
            "symbolic_timeout": self.symbolic_timeout,
            "symbolic_tolerance": self.symbolic_tolerance,
        }


def _package_version() -> str:
    try:
        return distribution_version("pdecert")
    except PackageNotFoundError:
        return "0+unknown"


def _select_records(
    records: Sequence[Mapping[str, object]],
    requested: Sequence[str] | None,
) -> list[Mapping[str, object]]:
    if requested is None:
        return list(records)
    if isinstance(requested, (str, bytes)):
        raise AtlasEvaluationError("record_ids must be a sequence of record identifiers")
    identifiers = list(requested)
    if not identifiers:
        raise AtlasEvaluationError("record_ids must not be empty when provided")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise AtlasEvaluationError("record_ids must contain non-empty strings")
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise AtlasEvaluationError("duplicate record id(s): " + ", ".join(duplicates))
    by_id = {record["id"]: record for record in records}
    unknown = sorted(set(identifiers) - set(by_id))
    if unknown:
        raise AtlasEvaluationError("unknown Atlas record id(s): " + ", ".join(unknown))
    return [by_id[identifier] for identifier in identifiers]


def _evaluate_record(
    record: Mapping[str, object],
    options: AtlasEvaluationOptions,
) -> tuple[dict[str, object], str | None]:
    record_id = str(record["id"])
    artifact_type = record["artifact_type"]
    try:
        template = template_from_dict(record["template"])
        artifact = record["artifact"]
        if not isinstance(artifact, Mapping):
            raise AtlasEvaluationError(f"record {record_id!r} artifact must be an object")

        if artifact_type == "symbolic_expression":
            fields = artifact.get("fields")
            if not isinstance(fields, Mapping):
                raise AtlasEvaluationError(
                    f"record {record_id!r} symbolic fields must be an object"
                )
            case = bind_symbolic_candidate(template, fields)
            report = verify(
                case.problem,
                SymbolicCandidate.from_expressions(case.candidate_fields),
                tolerance=options.symbolic_tolerance,
                samples_per_axis=options.samples_per_axis,
                symbolic_timeout=options.symbolic_timeout,
                max_expression_ops=options.max_expression_ops,
            )
            evaluator = "pdecert_symbolic"
            torch_version = None
        elif artifact_type == "callable_model":
            frozen = frozen_callable_from_dict(artifact)
            candidate = materialize_frozen_callable(frozen)
            report = verify_artifact(
                compile_autodiff_problem(template),
                candidate,
                tolerance=options.callable_tolerance,
                samples_per_axis=options.samples_per_axis,
            )
            evaluator = "pdecert_autodiff"
            import torch

            torch_version = torch.__version__
        else:
            raise AtlasEvaluationError(
                f"record {record_id!r} uses unsupported artifact type {artifact_type!r}"
            )
    except (CorpusError, FrozenCallableError, OperatorCompileError, TemplateError) as error:
        raise AtlasEvaluationError(f"record {record_id!r}: {error}") from error
    except RuntimeError as error:
        if "autodiff" in str(error):
            raise AtlasEvaluationError(
                f"record {record_id!r}: callable evaluation requires the 'autodiff' extra"
            ) from error
        raise

    return (
        {
            "artifact_id": artifact["artifact_id"],
            "artifact_type": artifact_type,
            "evaluator": evaluator,
            "problem_id": record["problem_id"],
            "record_id": record_id,
            "report": report.to_dict(),
        },
        torch_version,
    )


def evaluate_cross_artifact_atlas(
    path: str | Path,
    *,
    record_ids: Sequence[str] | None = None,
    options: AtlasEvaluationOptions | None = None,
) -> dict[str, object]:
    """Evaluate selected Atlas v2 records without aggregating their decisions.

    The Atlas is fully validated before any callable is materialized. Exact
    symbolic evidence and empirical callable evidence remain confined to their
    own records. The returned document intentionally has no overall status.
    """

    selected_options = options or AtlasEvaluationOptions()
    if not isinstance(selected_options, AtlasEvaluationOptions):
        raise TypeError("options must be an AtlasEvaluationOptions object")
    try:
        atlas = load_cross_artifact_atlas(path)
    except (OSError, CorpusError) as error:
        raise AtlasEvaluationError(str(error)) from error
    if atlas["atlas_version"] != CROSS_ARTIFACT_ATLAS_VERSION:
        raise AtlasEvaluationError(f"expected Atlas version {CROSS_ARTIFACT_ATLAS_VERSION}")
    records = _select_records(atlas["records"], record_ids)
    if not records:
        raise AtlasEvaluationError("Atlas contains no records to evaluate")

    evaluations: list[dict[str, object]] = []
    torch_version: str | None = None
    for record in records:
        evaluation, observed_torch_version = _evaluate_record(record, selected_options)
        evaluations.append(evaluation)
        if observed_torch_version is not None:
            torch_version = observed_torch_version

    runtime = {
        "pdecert_version": _package_version(),
        "python_version": platform.python_version(),
        "sympy_version": sympy.__version__,
        **({"torch_version": torch_version} if torch_version is not None else {}),
    }
    return {
        "atlas": {
            "atlas_version": atlas["atlas_version"],
            "name": atlas["name"],
            "sha256": cross_artifact_atlas_sha256(atlas),
        },
        "evaluation_version": ATLAS_EVALUATION_VERSION,
        "evidence_policy": "per_record_no_aggregation",
        "options": selected_options.to_dict(),
        "records": evaluations,
        "runtime": runtime,
    }
