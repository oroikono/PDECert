"""Load saved baseline diagnostics without treating validation as verification."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from .atlas_baselines import (
    AtlasBaselineError,
    BaselineOutcome,
    BaselineResult,
    BaselineWitness,
    DirectSympyBaseline,
    FixedCollocationBaseline,
    SymbolicBaselineCheck,
    SymbolicBaselineResult,
    _adapter_metadata,
)


ATLAS_BASELINE_MAX_BYTES = 64 * 1024 * 1024
_IDENTITY_FIELDS = {"artifact_id", "artifact_type", "problem_id", "record_id"}
_EMPIRICAL_FIELDS = {
    "outcome",
    "evidence_kind",
    "evidence_level",
    "evaluations",
    "max_absolute_residual",
    "witness",
    "reason",
}
_CHECK_FIELDS = {
    "obligation_id",
    "constraint",
    "constraint_source",
    "outcome",
    "materialized_residual",
    "simplified_residual",
    "reason",
    "evidence_kind",
    "evidence_level",
}


def _error(path: str, message: str) -> AtlasBaselineError:
    return AtlasBaselineError(f"{path}: {message}")


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _error(path, "must be an object")
    return value


def _fields(value: dict[str, object], path: str, expected: set[str]) -> None:
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing:
        raise _error(path, f"missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise _error(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a non-empty string")
    return value


def _json_value(value: object, path: str) -> object:
    """Copy strict JSON without coercing keys or integer witness coordinates."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise _error(path, "object keys must be strings")
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    raise _error(path, "must contain only finite, strict JSON values")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metadata(value: object, report_version: int) -> dict[str, object]:
    metadata = _object(value, "$.adapter")
    _fields(
        metadata,
        "$.adapter",
        {
            "accepted_artifact_types",
            "accepted_solution_semantics",
            "configuration",
            "id",
            "version",
        },
    )
    adapter_id = _text(metadata["id"], "$.adapter.id")
    if type(metadata["version"]) is not int or metadata["version"] < 1:
        raise _error("$.adapter.version", "must be a positive integer")
    for name in ("accepted_artifact_types", "accepted_solution_semantics"):
        values = metadata[name]
        if not isinstance(values, list) or not values:
            raise _error(f"$.adapter.{name}", "must be a non-empty array")
        for index, item in enumerate(values):
            _text(item, f"$.adapter.{name}[{index}]")
        if len(set(values)) != len(values):
            raise _error(f"$.adapter.{name}", "must not contain duplicates")
    configuration = _object(metadata["configuration"], "$.adapter.configuration")
    try:
        if adapter_id == "fixed_collocation":
            expected = FixedCollocationBaseline(
                decimal_precision=configuration.get("decimal_precision"),
                points_per_axis=configuration.get("points_per_axis"),
                tolerance=configuration.get("tolerance"),
            )
            expected_version = 1
        elif adapter_id == "direct_sympy":
            expected = DirectSympyBaseline(
                symbolic_timeout=configuration.get("symbolic_timeout_seconds"),
                max_expression_ops=configuration.get("max_expression_ops"),
            )
            expected_version = 2
        else:
            return metadata
    except (TypeError, ValueError) as error:
        raise _error("$.adapter.configuration", str(error)) from error
    if report_version != expected_version or _canonical(metadata) != _canonical(
        _adapter_metadata(expected)
    ):
        raise _error("$.adapter", f"{adapter_id} metadata/version must match its reserved contract")
    return metadata


def _empirical(record: dict[str, object], path: str) -> dict[str, object]:
    _fields(record, path, _IDENTITY_FIELDS | _EMPIRICAL_FIELDS)
    witness = None
    if record["witness"] is not None:
        value = _object(record["witness"], f"{path}.witness")
        _fields(
            value,
            f"{path}.witness",
            {
                "absolute_residual",
                "constraint",
                "constraint_source",
                "sampled_inputs",
            },
        )
        witness = BaselineWitness(**value)
    result = BaselineResult(
        outcome=BaselineOutcome(record["outcome"]),
        evidence_kind=record["evidence_kind"],
        evidence_level=record["evidence_level"],
        evaluations=record["evaluations"],
        max_absolute_residual=record["max_absolute_residual"],
        witness=witness,
        reason=record["reason"],
    )
    return result.to_dict()


def _symbolic(record: dict[str, object], path: str) -> dict[str, object]:
    _fields(record, path, _IDENTITY_FIELDS | {"checks", "outcome", "result_type", "scope"})
    if record["artifact_type"] != "symbolic_expression":
        raise _error(path, "symbolic checks require a symbolic artifact")
    values = record["checks"]
    if not isinstance(values, list) or not values:
        raise _error(f"{path}.checks", "must be a non-empty array")
    checks = []
    for index, value in enumerate(values):
        check_path = f"{path}.checks[{index}]"
        check = _object(value, check_path)
        _fields(check, check_path, _CHECK_FIELDS)
        typed = SymbolicBaselineCheck(
            **{
                key: item
                for key, item in check.items()
                if key not in {"evidence_kind", "evidence_level"}
            }
        )
        if check != typed.to_dict():
            raise _error(check_path, "evidence fields must match the symbolic outcome")
        checks.append(typed)
    result = SymbolicBaselineResult(tuple(checks)).to_dict()
    if result != {key: record[key] for key in result}:
        raise _error(path, "symbolic scope and aggregate outcome must match the checks")
    return result


def validate_atlas_baseline_report(value: object) -> dict[str, object]:
    """Check transport and internal consistency, not the source Atlas or mathematics.

    Return a defensive strict-JSON copy. Saved version-1 empirical and version-2
    symbolic/empirical reports retain their separate evidence vocabularies.
    """

    try:
        payload = _object(_json_value(value, "$"), "$")
        _fields(
            payload,
            "$",
            {
                "adapter",
                "atlas",
                "baseline_report_version",
                "evidence_policy",
                "records",
                "runtime",
            },
        )
        version = payload["baseline_report_version"]
        if type(version) is not int or version not in (1, 2):
            raise _error("$.baseline_report_version", "expected integer 1 or 2")
        policy = {
            1: "method_specific_empirical_diagnostics_no_proof",
            2: "method_specific_obligation_diagnostics_no_global_verdict",
        }[version]
        if payload["evidence_policy"] != policy:
            raise _error("$.evidence_policy", f"expected {policy!r}")
        metadata = _metadata(payload["adapter"], version)
        atlas = _object(payload["atlas"], "$.atlas")
        _fields(atlas, "$.atlas", {"atlas_version", "name", "sha256"})
        if type(atlas["atlas_version"]) is not int or atlas["atlas_version"] != 2:
            raise _error("$.atlas.atlas_version", "expected integer 2")
        _text(atlas["name"], "$.atlas.name")
        digest = _text(atlas["sha256"], "$.atlas.sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise _error("$.atlas.sha256", "must be a lowercase SHA-256 digest")
        runtime = _object(payload["runtime"], "$.runtime")
        _fields(
            runtime,
            "$.runtime",
            {
                "mpmath_version",
                "pdecert_version",
                "python_version",
                "sympy_version",
            },
        )
        for key, item in runtime.items():
            _text(item, f"$.runtime.{key}")
        records = payload["records"]
        if not isinstance(records, list) or not records:
            raise _error("$.records", "must be a non-empty array")
        identifiers = set()
        for index, value_record in enumerate(records):
            path = f"$.records[{index}]"
            record = _object(value_record, path)
            for key in _IDENTITY_FIELDS:
                if key not in record:
                    raise _error(path, f"missing field: {key}")
                _text(record[key], f"{path}.{key}")
            if record["record_id"] in identifiers:
                raise _error(path, "duplicate record id")
            identifiers.add(record["record_id"])
            if record["artifact_type"] not in {"symbolic_expression", "callable_model"}:
                raise _error(f"{path}.artifact_type", "unsupported artifact type")
            symbolic = "result_type" in record
            if symbolic and version != 2:
                raise _error(path, "symbolic rows require report version 2")
            result = _symbolic(record, path) if symbolic else _empirical(record, path)
            unsupported = result["outcome"] == "unsupported"
            if (
                record["artifact_type"] not in metadata["accepted_artifact_types"]
                and not unsupported
            ):
                raise _error(path, "out-of-scope artifacts must abstain")
            if metadata["id"] == "direct_sympy" and not symbolic and not unsupported:
                raise _error(path, "direct_sympy requires symbolic checks or an unsupported row")
            if metadata["id"] == "fixed_collocation" and not unsupported:
                maximum = result["max_absolute_residual"]
                tolerance = metadata["configuration"]["tolerance"]
                if result["evaluations"] > metadata["configuration"]["max_evaluations"]:
                    raise _error(path, "evaluation count exceeds the declared limit")
                if (result["outcome"] == "pass" and maximum > tolerance) or (
                    result["outcome"] == "fail" and maximum != "infinity" and maximum <= tolerance
                ):
                    raise _error(path, "outcome contradicts the declared tolerance")
        # Check encodability too, including Python's integer-string size limit.
        _canonical(payload)
        return payload
    except RecursionError as error:
        raise AtlasBaselineError("report nesting exceeds the supported limit") from error
    except AtlasBaselineError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise AtlasBaselineError(f"invalid baseline report: {error}") from error


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def load_atlas_baseline_report(path: str | Path) -> dict[str, object]:
    """Load a bounded regular JSON file and check its saved-report consistency."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AtlasBaselineError(f"{source}: expected a regular file")
    try:
        with source.open("rb") as stream:
            content = stream.read(ATLAS_BASELINE_MAX_BYTES + 1)
        if len(content) > ATLAS_BASELINE_MAX_BYTES:
            raise AtlasBaselineError(f"{source}: exceeds the {ATLAS_BASELINE_MAX_BYTES}-byte limit")
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
    except AtlasBaselineError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise AtlasBaselineError(f"{source}: could not load report: {error}") from error
    return validate_atlas_baseline_report(value)


def atlas_baseline_report_sha256(value: object) -> str:
    """Hash canonical validated report content, not authenticated run provenance."""

    return hashlib.sha256(
        _canonical(validate_atlas_baseline_report(value)).encode("utf-8")
    ).hexdigest()
