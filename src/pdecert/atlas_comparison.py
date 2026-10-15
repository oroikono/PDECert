"""Join saved Atlas reports without merging their evidence or rerunning methods."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from .atlas_baseline_reports import (
    atlas_baseline_report_sha256,
    validate_atlas_baseline_report,
)
from .atlas_evaluation import atlas_evaluation_sha256, validate_atlas_evaluation
from .corpus import cross_artifact_atlas_sha256, load_cross_artifact_atlas


ATLAS_COMPARISON_VERSION = 1
ATLAS_BOUND_COMPARISON_VERSION = 2
_IDENTITY_FIELDS = ("record_id", "problem_id", "artifact_id", "artifact_type")


class AtlasComparisonError(ValueError):
    """Raised when saved reports do not satisfy the requested comparison scope."""


def _json_copy(value: object) -> object:
    """Copy strict JSON values without silently coercing object keys or tuples."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise AtlasComparisonError("reports must contain only finite, strict JSON values")


def _bind_source_atlas(comparison: dict[str, object], atlas_path: str | Path) -> None:
    """Check source identity and obligation references without replaying methods."""

    try:
        atlas = load_cross_artifact_atlas(atlas_path)
        identity = {
            "atlas_version": atlas["atlas_version"],
            "name": atlas["name"],
            "sha256": cross_artifact_atlas_sha256(atlas),
        }
    except (OSError, TypeError, ValueError, RecursionError, OverflowError) as error:
        raise AtlasComparisonError(f"invalid source Atlas: {error}") from error
    if comparison["atlas"] != identity:
        raise AtlasComparisonError(
            "declared Atlas identity does not match the supplied source Atlas"
        )

    source_records = {record["id"]: record for record in atlas["records"]}
    adapter = comparison["sources"]["baseline"]["adapter"]
    for row in comparison["records"]:
        record_id = row["record_id"]
        if record_id not in source_records:
            raise AtlasComparisonError(f"record {record_id!r} is not in the supplied source Atlas")
        source = source_records[record_id]
        expected_identity = {
            "record_id": source["id"],
            "problem_id": source["problem_id"],
            "artifact_id": source["artifact"]["artifact_id"],
            "artifact_type": source["artifact_type"],
        }
        for field in _IDENTITY_FIELDS:
            if row[field] != expected_identity[field]:
                raise AtlasComparisonError(
                    f"record {record_id!r}: {field} differs from source Atlas"
                )

        template = source["template"]
        constraints = [
            (f"{group}[{index}]", constraint["name"], constraint["expression"])
            for group in ("pde_residuals", "conditions")
            for index, constraint in enumerate(template[group])
        ]
        obligations = {
            f"constraint:{index}": name for index, (_, name, _) in enumerate(constraints)
        }
        if source["artifact_type"] == "symbolic_expression":
            obligations.update(
                {
                    f"domain:{field}:{variable}": f"{field} domain"
                    for field in template["field_names"]
                    for variable in template["variables"]
                }
            )
        report = row["pdecert"]["report"]
        for event in report["evidence_events"]:
            obligation_id = event["obligation_id"]
            if obligation_id not in obligations:
                raise AtlasComparisonError(
                    f"record {record_id!r}: unknown evaluation obligation {obligation_id!r}"
                )
            witness = event["witness"]
            if witness is not None and witness["constraint"] != obligations[obligation_id]:
                raise AtlasComparisonError(
                    f"record {record_id!r}: witness constraint does not match {obligation_id!r}"
                )
        if report["status"] == "PROVED":
            discharged = {
                event["obligation_id"]
                for event in report["evidence_events"]
                if event["outcome"] == "DISCHARGED"
            }
            missing = sorted(obligations.keys() - discharged)
            if missing:
                raise AtlasComparisonError(
                    f"record {record_id!r}: PROVED report lacks discharged obligations: {missing}"
                )

        baseline = row["baseline"]
        if baseline["outcome"] != "unsupported" and (
            template["solution_semantics"] not in adapter["accepted_solution_semantics"]
        ):
            raise AtlasComparisonError(
                f"record {record_id!r}: baseline does not accept the source solution semantics"
            )
        if "checks" in baseline:
            observed = [
                (check["obligation_id"], check["constraint"], check["constraint_source"])
                for check in baseline["checks"]
            ]
            if observed != constraints:
                raise AtlasComparisonError(
                    f"record {record_id!r}: symbolic baseline must cover every source obligation "
                    "in order, with matching names and expressions"
                )
        elif baseline["witness"] is not None:
            witness = baseline["witness"]
            if (witness["constraint"], witness["constraint_source"]) not in {
                (name, expression) for _, name, expression in constraints
            }:
                raise AtlasComparisonError(
                    f"record {record_id!r}: baseline witness does not match a source obligation"
                )

    comparison["comparison_version"] = ATLAS_BOUND_COMPARISON_VERSION
    comparison["comparison_scope"] = "source_atlas_identity_and_obligation_scope_only"


def compare_atlas_reports(
    evaluation: object, baseline: object, *, atlas_path: str | Path | None = None
) -> dict[str, object]:
    """Compare one saved PDECert evaluation and one saved baseline report.

    Both reports must declare the same Atlas and contain exactly the same
    nonempty set of records, including problem and artifact identities. Native
    results stay separate. Without atlas_path, version 1 checks declared identity
    only. Supplying an Atlas v2 directory also checks source identity, record
    membership and obligation scope, and emits version 2. A bad source is an
    error, never a fallback to version 1. Neither mode checks mathematical
    correctness or execution provenance.
    Source digests identify the validated, normalized documents, not file bytes.
    """

    try:
        evaluated = validate_atlas_evaluation(_json_copy(evaluation))
        measured = validate_atlas_baseline_report(baseline)
        evaluation_digest = atlas_evaluation_sha256(evaluated)
        baseline_digest = atlas_baseline_report_sha256(measured)
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as error:
        raise AtlasComparisonError(f"invalid source report: {error}") from error

    if evaluated["atlas"] != measured["atlas"]:
        raise AtlasComparisonError("reports must declare the same Atlas version, name and SHA-256")

    evaluation_rows = {row["record_id"]: row for row in evaluated["records"]}
    baseline_rows = {row["record_id"]: row for row in measured["records"]}
    if evaluation_rows.keys() != baseline_rows.keys():
        missing = sorted(evaluation_rows.keys() - baseline_rows.keys())
        extra = sorted(baseline_rows.keys() - evaluation_rows.keys())
        raise AtlasComparisonError(
            f"record sets differ: missing from baseline={missing}; extra in baseline={extra}"
        )

    records = []
    for record_id in sorted(evaluation_rows):
        evaluated_row, baseline_row = evaluation_rows[record_id], baseline_rows[record_id]
        for field in _IDENTITY_FIELDS:
            if evaluated_row[field] != baseline_row[field]:
                raise AtlasComparisonError(f"record {record_id!r}: {field} differs between reports")
        records.append(
            {
                **{field: evaluated_row[field] for field in _IDENTITY_FIELDS},
                "pdecert": {
                    "evaluator": evaluated_row["evaluator"],
                    "report": evaluated_row["report"],
                },
                "baseline": {
                    key: value for key, value in baseline_row.items() if key not in _IDENTITY_FIELDS
                },
            }
        )

    comparison = {
        "atlas": evaluated["atlas"],
        "comparison_scope": "declared_identity_and_native_evidence_only",
        "comparison_version": ATLAS_COMPARISON_VERSION,
        "evidence_policy": "per_record_no_aggregation",
        "records": records,
        "sources": {
            "evaluation": {
                "evaluation_version": evaluated["evaluation_version"],
                "evidence_policy": evaluated["evidence_policy"],
                "options": evaluated["options"],
                "report_sha256": evaluation_digest,
                "runtime": evaluated["runtime"],
            },
            "baseline": {
                "adapter": measured["adapter"],
                "baseline_report_version": measured["baseline_report_version"],
                "evidence_policy": measured["evidence_policy"],
                "report_sha256": baseline_digest,
                "runtime": measured["runtime"],
            },
        },
    }
    if atlas_path is not None:
        _bind_source_atlas(comparison, atlas_path)
    return comparison
