"""Evidence-preserving evaluation of typed Atlas v2 records."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
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
from .reports import ReportSchemaError, report_from_dict
from .templates import TemplateError, bind_symbolic_candidate, template_from_dict


ATLAS_EVALUATION_VERSION = 1
ATLAS_EVALUATION_SUMMARY_VERSION = 1
ATLAS_EVALUATION_MAX_BYTES = 64 * 1024 * 1024


class AtlasEvaluationError(ValueError):
    """Raised when a typed Atlas record cannot be evaluated under this contract."""


def _evaluation_error(path: str, message: str) -> AtlasEvaluationError:
    return AtlasEvaluationError(f"{path}: {message}")


def _evaluation_object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _evaluation_error(path, "must be an object")
    if not all(isinstance(key, str) for key in value):
        raise _evaluation_error(path, "object keys must be strings")
    return value


def _evaluation_fields(
    value: Mapping[str, object],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing:
        raise _evaluation_error(path, f"missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise _evaluation_error(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _evaluation_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _evaluation_error(path, "must be a non-empty string")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = item
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


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
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            try:
                normalized = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be finite and positive") from error
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, normalized)
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


def validate_atlas_evaluation(value: object) -> dict[str, object]:
    """Validate and normalize one version-1 typed Atlas evaluation document."""

    payload = _evaluation_object(value, "$")
    root_fields = {
        "atlas",
        "evaluation_version",
        "evidence_policy",
        "options",
        "records",
        "runtime",
    }
    _evaluation_fields(payload, "$", required=root_fields)
    version = payload["evaluation_version"]
    if isinstance(version, bool) or version != ATLAS_EVALUATION_VERSION:
        raise _evaluation_error(
            "$.evaluation_version",
            f"expected {ATLAS_EVALUATION_VERSION}",
        )
    if payload["evidence_policy"] != "per_record_no_aggregation":
        raise _evaluation_error(
            "$.evidence_policy",
            "expected 'per_record_no_aggregation'",
        )

    atlas = _evaluation_object(payload["atlas"], "$.atlas")
    _evaluation_fields(
        atlas,
        "$.atlas",
        required={"atlas_version", "name", "sha256"},
    )
    atlas_version = atlas["atlas_version"]
    if isinstance(atlas_version, bool) or atlas_version != CROSS_ARTIFACT_ATLAS_VERSION:
        raise _evaluation_error(
            "$.atlas.atlas_version",
            f"expected {CROSS_ARTIFACT_ATLAS_VERSION}",
        )
    atlas_name = _evaluation_text(atlas["name"], "$.atlas.name")
    atlas_digest = _evaluation_text(atlas["sha256"], "$.atlas.sha256")
    if len(atlas_digest) != 64 or any(
        character not in "0123456789abcdef" for character in atlas_digest
    ):
        raise _evaluation_error("$.atlas.sha256", "must be a lowercase SHA-256 digest")

    options = _evaluation_object(payload["options"], "$.options")
    option_fields = {
        "callable_tolerance",
        "max_expression_ops",
        "samples_per_axis",
        "symbolic_timeout",
        "symbolic_tolerance",
    }
    _evaluation_fields(options, "$.options", required=option_fields)
    try:
        normalized_options = AtlasEvaluationOptions(**options).to_dict()
    except (TypeError, ValueError) as error:
        raise _evaluation_error("$.options", str(error)) from error

    runtime = _evaluation_object(payload["runtime"], "$.runtime")
    _evaluation_fields(
        runtime,
        "$.runtime",
        required={"pdecert_version", "python_version", "sympy_version"},
        optional={"torch_version"},
    )
    normalized_runtime = {
        name: _evaluation_text(item, f"$.runtime.{name}") for name, item in runtime.items()
    }

    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise _evaluation_error("$.records", "must be a non-empty array")
    normalized_records: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for index, value_record in enumerate(records):
        path = f"$.records[{index}]"
        record = _evaluation_object(value_record, path)
        _evaluation_fields(
            record,
            path,
            required={
                "artifact_id",
                "artifact_type",
                "evaluator",
                "problem_id",
                "record_id",
                "report",
            },
        )
        record_id = _evaluation_text(record["record_id"], f"{path}.record_id")
        if record_id in identifiers:
            raise _evaluation_error(f"{path}.record_id", f"duplicate record id {record_id!r}")
        identifiers.add(record_id)
        artifact_type = record["artifact_type"]
        evaluator = record["evaluator"]
        expected_evaluator = {
            "symbolic_expression": "pdecert_symbolic",
            "callable_model": "pdecert_autodiff",
        }.get(artifact_type)
        if expected_evaluator is None:
            raise _evaluation_error(
                f"{path}.artifact_type",
                f"unsupported value {artifact_type!r}",
            )
        if evaluator != expected_evaluator:
            raise _evaluation_error(
                f"{path}.evaluator",
                f"expected {expected_evaluator!r} for {artifact_type!r}",
            )
        try:
            report = report_from_dict(record["report"])
        except (OverflowError, ReportSchemaError) as error:
            raise _evaluation_error(f"{path}.report", str(error)) from error
        normalized_report = report.to_dict()
        if artifact_type == "callable_model":
            if normalized_report["status"] == "PROVED":
                raise _evaluation_error(
                    f"{path}.report.status",
                    "callable evaluations cannot be PROVED",
                )
            if normalized_report["decision_evidence"] not in {None, "EMPIRICAL"}:
                raise _evaluation_error(
                    f"{path}.report.decision_evidence",
                    "callable evaluations may carry only empirical decision evidence",
                )
            if normalized_report["exact_checks"]:
                raise _evaluation_error(
                    f"{path}.report.exact_checks",
                    "callable evaluations cannot carry symbolic exact checks",
                )
            allowed_event_kinds = {
                "ABSTENTION",
                "EMPIRICAL_COUNTEREXAMPLE",
                "EMPIRICAL_PASS",
            }
            for event_index, event in enumerate(normalized_report["evidence_events"]):
                if event["kind"] not in allowed_event_kinds:
                    raise _evaluation_error(
                        f"{path}.report.evidence_events[{event_index}].kind",
                        "callable evaluations may carry only empirical events or abstentions",
                    )

        normalized_records.append(
            {
                "artifact_id": _evaluation_text(record["artifact_id"], f"{path}.artifact_id"),
                "artifact_type": artifact_type,
                "evaluator": evaluator,
                "problem_id": _evaluation_text(record["problem_id"], f"{path}.problem_id"),
                "record_id": record_id,
                "report": normalized_report,
            }
        )

    if any(record["artifact_type"] == "callable_model" for record in normalized_records) and (
        "torch_version" not in normalized_runtime
    ):
        raise _evaluation_error(
            "$.runtime.torch_version",
            "is required when callable records are present",
        )

    return {
        "atlas": {
            "atlas_version": CROSS_ARTIFACT_ATLAS_VERSION,
            "name": atlas_name,
            "sha256": atlas_digest,
        },
        "evaluation_version": ATLAS_EVALUATION_VERSION,
        "evidence_policy": "per_record_no_aggregation",
        "options": normalized_options,
        "records": normalized_records,
        "runtime": normalized_runtime,
    }


def load_atlas_evaluation(path: str | Path) -> dict[str, object]:
    """Load and validate one version-1 typed Atlas evaluation JSON file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AtlasEvaluationError(f"{source}: expected a regular file")
    try:
        if source.stat().st_size > ATLAS_EVALUATION_MAX_BYTES:
            raise AtlasEvaluationError(
                f"{source}: file exceeds the {ATLAS_EVALUATION_MAX_BYTES}-byte limit"
            )
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError) as error:
        raise AtlasEvaluationError(f"{source}: could not read file: {error}") from error
    except RecursionError as error:
        raise AtlasEvaluationError(f"{source}: JSON nesting exceeds the decoder limit") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise AtlasEvaluationError(f"{source}: invalid JSON: {error}") from error
    return validate_atlas_evaluation(value)


def atlas_evaluation_sha256(value: object) -> str:
    """Hash one validated Atlas evaluation using canonical strict JSON."""

    normalized = validate_atlas_evaluation(value)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize_atlas_evaluation(value: object) -> dict[str, object]:
    """Create a descriptive cross-artifact summary without labels or aggregation."""

    evaluation = validate_atlas_evaluation(value)
    records = evaluation["records"]
    artifact_types = Counter(record["artifact_type"] for record in records)
    evaluators = Counter(record["evaluator"] for record in records)
    statuses = Counter(record["report"]["status"] for record in records)
    decision_evidence = Counter(
        record["report"]["decision_evidence"] or "NONE" for record in records
    )

    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(record["problem_id"], []).append(record)
    problems: list[dict[str, object]] = []
    for problem_id in sorted(grouped):
        problem_records = sorted(grouped[problem_id], key=lambda record: record["record_id"])
        problems.append(
            {
                "artifact_types": sorted({record["artifact_type"] for record in problem_records}),
                "problem_id": problem_id,
                "records": [
                    {
                        "artifact_id": record["artifact_id"],
                        "artifact_type": record["artifact_type"],
                        "decision_evidence": record["report"]["decision_evidence"],
                        "evaluator": record["evaluator"],
                        "record_id": record["record_id"],
                        "status": record["report"]["status"],
                        "witness_present": record["report"]["witness"] is not None,
                    }
                    for record in problem_records
                ],
                "statuses": dict(
                    sorted(
                        Counter(record["report"]["status"] for record in problem_records).items()
                    )
                ),
            }
        )

    return {
        "coverage": {
            "artifact_types": dict(sorted(artifact_types.items())),
            "cross_artifact_problems": sum(
                len(problem["artifact_types"]) > 1 for problem in problems
            ),
            "evaluators": dict(sorted(evaluators.items())),
            "problems": len(problems),
            "records": len(records),
        },
        "evidence_policy": "descriptive_per_record_no_truth_labels",
        "outcomes": {
            "decision_evidence": dict(sorted(decision_evidence.items())),
            "statuses": dict(sorted(statuses.items())),
            "witnesses": sum(record["report"]["witness"] is not None for record in records),
        },
        "problems": problems,
        "source": {
            "atlas_name": evaluation["atlas"]["name"],
            "atlas_sha256": evaluation["atlas"]["sha256"],
            "evaluation_sha256": atlas_evaluation_sha256(evaluation),
            "evaluation_version": ATLAS_EVALUATION_VERSION,
        },
        "summary_version": ATLAS_EVALUATION_SUMMARY_VERSION,
    }
