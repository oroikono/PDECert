"""Join saved Atlas reports without merging their evidence or rerunning methods."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .atlas_baseline_reports import (
    atlas_baseline_report_sha256,
    validate_atlas_baseline_report,
)
from .atlas_evaluation import atlas_evaluation_sha256, validate_atlas_evaluation


ATLAS_COMPARISON_VERSION = 1
_IDENTITY_FIELDS = ("record_id", "problem_id", "artifact_id", "artifact_type")


class AtlasComparisonError(ValueError):
    """Raised when saved reports cannot be compared under the same declared identity."""


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


def compare_atlas_reports(evaluation: object, baseline: object) -> dict[str, object]:
    """Compare one saved PDECert evaluation and one saved baseline report.

    Both reports must declare the same Atlas and contain exactly the same
    nonempty set of records, including problem and artifact identities. Native
    results stay separate. This checks report consistency and declared identity,
    not the source Atlas, mathematical correctness, or execution provenance.
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

    return {
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
