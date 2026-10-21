"""Read supplied field samples without loading or executing a model."""

from __future__ import annotations

import json
from pathlib import Path

from .reference_fields import ReferenceComparisonError, compare_reference_fields

REFERENCE_INPUT_VERSION = 1
REFERENCE_INPUT_MAX_BYTES = 64 * 1024 * 1024

_INPUT_FIELDS = frozenset(
    {
        "reference_input_version",
        "coordinates",
        "candidate",
        "reference",
        "problem_id",
        "candidate_id",
        "reference_id",
        "reference_description",
        "reference_uncertainty",
        "sampling_description",
    }
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ReferenceComparisonError(f"duplicate JSON field: {name!r}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReferenceComparisonError(f"nonfinite JSON constant: {value}")


def compare_reference_file(path: str | Path) -> dict[str, object]:
    """Compare a version-1 UTF-8 JSON sample file using the existing metrics.

    The file supplies aligned numeric columns and reference/sampling metadata.
    Its envelope version is not part of the normalized-input digest. Reading
    is byte-bounded, but is not a memory sandbox or protection against concurrent
    filesystem changes. File access errors propagate as ``OSError``; malformed
    or unsupported data raises ``ReferenceComparisonError``.
    """
    source = Path(path)
    if not source.is_file():
        raise ReferenceComparisonError(f"{source}: expected a regular input file")
    with source.open("rb") as stream:
        raw = stream.read(REFERENCE_INPUT_MAX_BYTES + 1)
    if len(raw) > REFERENCE_INPUT_MAX_BYTES:
        raise ReferenceComparisonError(
            f"reference input exceeds the {REFERENCE_INPUT_MAX_BYTES}-byte limit"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReferenceComparisonError(f"invalid reference input JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ReferenceComparisonError("reference input must be a JSON object")
    missing = _INPUT_FIELDS - payload.keys()
    unexpected = payload.keys() - _INPUT_FIELDS
    if missing or unexpected:
        raise ReferenceComparisonError(
            f"reference input fields: missing {sorted(missing)}; unexpected {sorted(unexpected)}"
        )
    version = payload.pop("reference_input_version")
    if type(version) is not int or version != REFERENCE_INPUT_VERSION:
        raise ReferenceComparisonError("reference_input_version must be the integer 1")
    try:
        for value in payload.values():
            if isinstance(value, str):
                value.encode("utf-8")
        for group in ("coordinates", "candidate", "reference"):
            if isinstance(payload[group], dict):
                for name in payload[group]:
                    name.encode("utf-8")
        return compare_reference_fields(**payload)
    except UnicodeError as error:
        raise ReferenceComparisonError("reference input strings must be valid UTF-8") from error
