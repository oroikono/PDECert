"""Empirical comparisons of aligned, already evaluated field values."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version

REFERENCE_COMPARISON_VERSION = 1
REFERENCE_COMPARISON_MAX_VALUES = 1_000_000


class ReferenceComparisonError(ValueError):
    """Input cannot be compared within the finite binary64 contract."""


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceComparisonError(f"{path}: expected a nonblank string")
    return value


def _columns(value: object, path: str) -> dict[str, Sequence]:
    if not isinstance(value, Mapping) or not value:
        raise ReferenceComparisonError(f"{path}: expected a nonempty mapping of columns")
    result = {}
    for name, column in value.items():
        _text(name, f"{path} column name")
        if isinstance(column, (str, bytes)) or not isinstance(column, Sequence):
            raise ReferenceComparisonError(f"{path}.{name}: expected a flat numeric sequence")
        result[name] = column
    return result


def _numbers(columns: Mapping[str, Sequence], path: str) -> dict[str, list[float]]:
    result = {}
    for name in sorted(columns):
        values = []
        for index, value in enumerate(columns[name]):
            location = f"{path}.{name}[{index}]"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReferenceComparisonError(f"{location}: expected a real int or float")
            try:
                number = float(value)
            except OverflowError as error:
                raise ReferenceComparisonError(f"{location}: value exceeds binary64") from error
            if not math.isfinite(number):
                raise ReferenceComparisonError(f"{location}: expected a finite value")
            if isinstance(value, int) and int(number) != value:
                raise ReferenceComparisonError(f"{location}: integer loses precision in binary64")
            values.append(0.0 if number == 0 else number)
        result[name] = values
    return result


def _relative_l2(errors: list[float], reference: list[float]) -> tuple[float | None, str | None]:
    error_scale = max(errors)
    reference_scale = max(abs(value) for value in reference)
    if reference_scale == 0:
        return None, "zero_reference_norm"
    if error_scale == 0:
        return 0.0, None
    error_squares = math.fsum((value / error_scale) ** 2 for value in errors)
    reference_squares = math.fsum((value / reference_scale) ** 2 for value in reference)
    error_mantissa, error_exponent = math.frexp(error_scale)
    reference_mantissa, reference_exponent = math.frexp(reference_scale)
    factor = error_mantissa / reference_mantissa * math.sqrt(error_squares / reference_squares)
    try:
        relative = math.ldexp(factor, error_exponent - reference_exponent)
    except OverflowError:
        return None, "binary64_overflow"
    if relative == 0:
        return None, "binary64_underflow"
    return relative, None


def compare_reference_fields(
    coordinates: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
    *,
    problem_id: str,
    candidate_id: str,
    reference_id: str,
    reference_description: str,
    reference_uncertainty: str,
    sampling_description: str,
) -> dict[str, object]:
    """Compare finite, aligned scalar columns without issuing a PDE verdict.

    Each row receives equal weight, including duplicate points. Inputs must be
    flat Python numeric sequences; convert framework arrays explicitly. The
    caller owns alignment, units, reference quality and problem identity. No
    model, derivative, interpolation or problem-template evaluation is run.

    The returned report includes per-field RMSE, discrete relative L2, the
    largest sampled absolute difference, and normalized-input identity. It
    contains no proof, tolerance decision, or true-solution error bound.
    """
    for name, value in (
        ("problem_id", problem_id),
        ("candidate_id", candidate_id),
        ("reference_id", reference_id),
        ("reference_description", reference_description),
        ("reference_uncertainty", reference_uncertainty),
        ("sampling_description", sampling_description),
    ):
        _text(value, name)
    columns = {
        "coordinates": _columns(coordinates, "coordinates"),
        "candidate": _columns(candidate, "candidate"),
        "reference": _columns(reference, "reference"),
    }
    if set(columns["candidate"]) != set(columns["reference"]):
        raise ReferenceComparisonError("candidate and reference field names must match")
    count = len(next(iter(columns["coordinates"].values())))
    if count == 0:
        raise ReferenceComparisonError("coordinates: expected at least one sample")
    if count * sum(len(group) for group in columns.values()) > REFERENCE_COMPARISON_MAX_VALUES:
        raise ReferenceComparisonError("sample columns exceed the maximum numeric value budget")
    for path, group in columns.items():
        for name, column in group.items():
            if len(column) != count:
                raise ReferenceComparisonError(f"{path}.{name}: expected {count} aligned samples")
    normalized = {path: _numbers(group, path) for path, group in columns.items()}
    fields = []
    for name, predictions in normalized["candidate"].items():
        references = normalized["reference"][name]
        errors = [abs(predicted - actual) for predicted, actual in zip(predictions, references)]
        if any(not math.isfinite(error) for error in errors):
            raise ReferenceComparisonError(f"field {name!r}: absolute difference exceeds binary64")
        maximum = max(errors)
        rmse = (
            maximum * math.sqrt(math.fsum((error / maximum) ** 2 for error in errors) / count)
            if maximum
            else 0.0
        )
        rmse_reason = "binary64_underflow" if maximum and rmse == 0 else None
        relative, relative_reason = _relative_l2(errors, references)
        index = errors.index(maximum)
        fields.append(
            {
                "field": name,
                "rmse": None if rmse_reason else rmse,
                "rmse_unavailable_reason": rmse_reason,
                "relative_l2": relative,
                "relative_l2_unavailable_reason": relative_reason,
                "max_absolute_error": maximum,
                "max_error_sample": {
                    "index": index,
                    "coordinates": {
                        coordinate: values[index]
                        for coordinate, values in normalized["coordinates"].items()
                    },
                    "candidate_value": predictions[index],
                    "reference_value": references[index],
                },
            }
        )
    metadata = {
        "problem_id": problem_id,
        "candidate_id": candidate_id,
        "reference": {
            "id": reference_id,
            "description": reference_description,
            "uncertainty": reference_uncertainty,
            "assessment": "caller_supplied_not_verified",
        },
        "sampling_description": sampling_description,
    }
    identity = json.dumps(
        {"metadata": metadata, "samples": normalized},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        package_version = version("pdecert")
    except PackageNotFoundError:
        package_version = "0+unknown"
    return {
        "reference_comparison_version": REFERENCE_COMPARISON_VERSION,
        "evidence_level": "EMPIRICAL",
        "scope": "supplied_field_values_at_supplied_points_only",
        **metadata,
        "sample_count": count,
        "coordinate_names": list(normalized["coordinates"]),
        "input_sha256": hashlib.sha256(identity).hexdigest(),
        "input_identity_scope": "normalized_binary64_samples_and_declared_metadata_only",
        "configuration": {
            "arithmetic": "python_binary64",
            "reduction": "equal_weight_per_row_duplicates_retained",
            "relative_norm": "per_field_discrete_l2",
            "units": "as_supplied_no_conversion",
            "max_values": REFERENCE_COMPARISON_MAX_VALUES,
        },
        "fields": fields,
        "runtime": {
            "python_version": platform.python_version(),
            "pdecert_version": package_version,
        },
    }
