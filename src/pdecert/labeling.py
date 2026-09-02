"""Human-review import for candidate-corpus annotations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from .corpus import (
    CROSS_ARTIFACT_ATLAS_VERSION,
    FAILURE_MODES,
    REVIEW_BASIS_KINDS,
    VERDICTS,
    validate_corpus,
)


REVIEW_VERSION = 1
CROSS_ARTIFACT_REVIEW_VERSION = 2
_BASES_BY_DECISION = {
    "symbolic_expression": {
        "valid": {"manual_derivation", "rigorous_external_certificate"},
        "invalid": {
            "independent_counterexample",
            "manual_derivation",
            "rigorous_external_certificate",
        },
        "unclear": {"scope_assessment"},
    },
    "callable_model": {
        "valid": {"rigorous_external_certificate"},
        "invalid": {"independent_counterexample", "rigorous_external_certificate"},
        "unclear": {"scope_assessment"},
    },
}


class ReviewError(ValueError):
    """Raised when a review cannot be applied to a corpus."""


def _error(path: str, message: str) -> ReviewError:
    return ReviewError(f"{path}: {message}")


def review_source_sha256(source: object) -> str:
    """Return the canonical digest that binds a review to loaded source data."""

    encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def allowed_review_bases(artifact_type: str, verdict: str) -> tuple[str, ...]:
    """Return review bases that can support one typed-record decision."""

    try:
        return tuple(sorted(_BASES_BY_DECISION[artifact_type][verdict]))
    except KeyError as error:
        raise ReviewError(
            f"unsupported typed review decision: {artifact_type}/{verdict}"
        ) from error


def _validate_decision(
    value: object,
    path: str,
    *,
    artifact_type: str | None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected an object")
    fields = {"failure_modes", "id", "rationale", "verdict"}
    if artifact_type is not None:
        fields.add("basis")
    if set(value) != fields:
        raise _error(path, f"expected {', '.join(sorted(fields))}")
    record_id = value["id"]
    if not isinstance(record_id, str) or not record_id:
        raise _error(f"{path}.id", "expected a non-empty string")
    verdict = value["verdict"]
    if verdict not in VERDICTS:
        raise _error(f"{path}.verdict", f"expected one of: {', '.join(sorted(VERDICTS))}")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise _error(f"{path}.rationale", "expected a non-empty string")
    failure_modes = value["failure_modes"]
    if not isinstance(failure_modes, list) or any(
        not isinstance(item, str) for item in failure_modes
    ):
        raise _error(f"{path}.failure_modes", "expected a list of strings")
    unknown_modes = set(failure_modes) - FAILURE_MODES
    if unknown_modes:
        raise _error(
            f"{path}.failure_modes",
            f"unsupported failure mode(s): {', '.join(sorted(unknown_modes))}",
        )
    if len(set(failure_modes)) != len(failure_modes):
        raise _error(f"{path}.failure_modes", "failure modes must be unique")
    if verdict == "invalid" and not failure_modes:
        raise _error(f"{path}.failure_modes", "invalid verdicts require a failure mode")
    if verdict != "invalid" and failure_modes:
        raise _error(f"{path}.failure_modes", "only invalid verdicts may have failure modes")

    if artifact_type is not None:
        basis = value["basis"]
        if not isinstance(basis, Mapping) or set(basis) != {"description", "kind"}:
            raise _error(f"{path}.basis", "expected exactly kind and description")
        kind = basis["kind"]
        if kind not in REVIEW_BASIS_KINDS:
            raise _error(
                f"{path}.basis.kind",
                f"expected one of: {', '.join(sorted(REVIEW_BASIS_KINDS))}",
            )
        description = basis["description"]
        if not isinstance(description, str) or not description.strip():
            raise _error(f"{path}.basis.description", "expected a non-empty string")
        allowed = allowed_review_bases(artifact_type, verdict)
        if kind not in allowed:
            raise _error(
                f"{path}.basis.kind",
                f"{kind!r} cannot support {verdict!r} for {artifact_type}; "
                f"expected one of: {', '.join(allowed)}",
            )
    return value


def apply_review(
    corpus: object,
    review: object,
    *,
    annotator: str,
    confirmed_independent_review: bool,
) -> dict[str, object]:
    """Apply one complete human review without mutating the input corpus."""

    if not confirmed_independent_review:
        raise ReviewError("independent human review must be explicitly confirmed")
    if not isinstance(annotator, str) or not annotator.strip():
        raise ReviewError("annotator must be a non-empty identifier")
    if not isinstance(corpus, Mapping):
        raise ReviewError("corpus must be an object")
    is_cross_artifact = corpus.get("atlas_version") == CROSS_ARTIFACT_ATLAS_VERSION
    if not is_cross_artifact:
        validate_corpus(corpus)
    if not isinstance(review, Mapping):
        raise ReviewError("review must be an object")
    expected_review_fields = (
        {"atlas_sha256", "records", "review_version"}
        if is_cross_artifact
        else {"records", "review_version"}
    )
    if set(review) != expected_review_fields:
        raise ReviewError(
            "review must contain exactly " + ", ".join(sorted(expected_review_fields))
        )
    expected_version = CROSS_ARTIFACT_REVIEW_VERSION if is_cross_artifact else REVIEW_VERSION
    if isinstance(review["review_version"], bool) or review["review_version"] != expected_version:
        raise ReviewError(f"review_version must be {expected_version}")
    if is_cross_artifact:
        expected_digest = review_source_sha256(corpus)
        if review["atlas_sha256"] != expected_digest:
            raise ReviewError("review atlas_sha256 does not match the source Atlas")
    review_records = review["records"]
    if not isinstance(review_records, list):
        raise ReviewError("records must be a list")

    corpus_records = corpus.get("records")
    if not isinstance(corpus_records, list):
        raise ReviewError("corpus records must be a list")
    artifact_types = {
        record["id"]: record["artifact_type"]
        for record in corpus_records
        if is_cross_artifact and isinstance(record, Mapping)
    }
    decisions: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(review_records):
        path = f"$.records[{index}]"
        record_id = value.get("id") if isinstance(value, Mapping) else None
        artifact_type = artifact_types.get(record_id) if is_cross_artifact else None
        decision = _validate_decision(value, path, artifact_type=artifact_type)
        record_id = decision["id"]
        if record_id in decisions:
            raise _error(f"{path}.id", f"duplicate review id: {record_id}")
        decisions[record_id] = decision

    corpus_ids = [record["id"] for record in corpus_records]
    missing = sorted(set(corpus_ids) - set(decisions))
    unknown = sorted(set(decisions) - set(corpus_ids))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ReviewError("review record IDs do not match corpus: " + "; ".join(details))

    labeled = copy.deepcopy(corpus)
    for record in labeled["records"]:
        if record["annotation"]["status"] != "pending":
            raise ReviewError(f"record is already annotated: {record['id']}")
        decision = decisions[record["id"]]
        record["annotation"] = {
            "status": "labeled",
            "verdict": decision["verdict"],
            "failure_modes": list(decision["failure_modes"]),
            "rationale": decision["rationale"],
            "annotators": [annotator],
            **({"review_basis": copy.deepcopy(decision["basis"])} if is_cross_artifact else {}),
        }
    if not is_cross_artifact:
        validate_corpus(labeled)
    return labeled
