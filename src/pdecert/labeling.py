"""Human-review import for candidate-corpus annotations."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from .corpus import FAILURE_MODES, VERDICTS, validate_corpus


REVIEW_VERSION = 1


class ReviewError(ValueError):
    """Raised when a review cannot be applied to a corpus."""


def _error(path: str, message: str) -> ReviewError:
    return ReviewError(f"{path}: {message}")


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
    validate_corpus(corpus)
    if not isinstance(corpus, Mapping):
        raise ReviewError("corpus must be an object")
    if not isinstance(review, Mapping):
        raise ReviewError("review must be an object")
    if set(review) != {"records", "review_version"}:
        raise ReviewError("review must contain exactly review_version and records")
    if isinstance(review["review_version"], bool) or review["review_version"] != REVIEW_VERSION:
        raise ReviewError(f"review_version must be {REVIEW_VERSION}")
    review_records = review["records"]
    if not isinstance(review_records, list):
        raise ReviewError("records must be a list")

    decisions: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(review_records):
        path = f"$.records[{index}]"
        if not isinstance(value, Mapping):
            raise _error(path, "expected an object")
        if set(value) != {"failure_modes", "id", "rationale", "verdict"}:
            raise _error(path, "expected id, verdict, failure_modes, and rationale")
        record_id = value["id"]
        if not isinstance(record_id, str) or not record_id:
            raise _error(f"{path}.id", "expected a non-empty string")
        if record_id in decisions:
            raise _error(f"{path}.id", f"duplicate review id: {record_id}")
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
        decisions[record_id] = value

    corpus_records = corpus["records"]
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
        }
    validate_corpus(labeled)
    return labeled
