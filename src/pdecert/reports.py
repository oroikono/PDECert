"""Versioned serialization for verification reports and evidence events."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from .core import (
    AGGREGATION_POLICY_VERSION,
    REPORT_VERSION,
    Report,
    Status,
)
from .evidence import (
    BoundEvidence,
    BoundType,
    EvidenceEvent,
    EvidenceKind,
    EvidenceLevel,
    EvidenceOutcome,
    Witness,
)


class ReportSchemaError(ValueError):
    """Raised when a serialized report violates the public report contract."""


_EnumType = TypeVar("_EnumType")


def _error(path: str, message: str) -> ReportSchemaError:
    return ReportSchemaError(f"{path}: {message}")


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be an object")
    if not all(isinstance(key, str) for key in value):
        raise _error(path, "object keys must be strings")
    return value


def _fields(
    value: Mapping[str, object],
    path: str,
    *,
    required: set[str],
) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise _error(path, f"missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise _error(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a non-empty string")
    return value


def _enum(enum_type: type[_EnumType], value: object, path: str) -> _EnumType:
    if not isinstance(value, str):
        raise _error(path, "must be a string")
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        raise _error(path, f"unsupported value {value!r}; expected one of: {allowed}") from error


def _finite_number(value: object, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _error(path, "must be a finite number")
    if nonnegative and numeric < 0:
        raise _error(path, "must be nonnegative")
    return numeric


def _string_map(value: object, path: str) -> dict[str, str]:
    mapping = _object(value, path)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        result[_string(key, f"{path}.<key>")] = _string(item, f"{path}.{key}")
    return result


def _witness_from_dict(value: object, path: str) -> Witness:
    payload = _object(value, path)
    required = {"constraint", "point", "residual", "reason"}
    _fields(payload, path, required=required)
    point_payload = _object(payload["point"], f"{path}.point")
    point: dict[str, float | str] = {}
    for name, coordinate in point_payload.items():
        key = _string(name, f"{path}.point.<key>")
        if isinstance(coordinate, str):
            point[key] = _string(coordinate, f"{path}.point.{key}")
        else:
            point[key] = _finite_number(coordinate, f"{path}.point.{key}")

    residual = payload["residual"]
    if isinstance(residual, str):
        parsed_residual: float | str = _string(residual, f"{path}.residual")
    else:
        parsed_residual = _finite_number(residual, f"{path}.residual", nonnegative=True)
    return Witness(
        constraint=_string(payload["constraint"], f"{path}.constraint"),
        point=point,
        residual=parsed_residual,
        reason=_string(payload["reason"], f"{path}.reason"),
    )


def _bound_from_dict(value: object, path: str) -> BoundEvidence:
    payload = _object(value, path)
    required = {
        "bound_type",
        "quantity",
        "upper_bound",
        "norm",
        "scope",
        "assumptions",
        "constants",
    }
    _fields(payload, path, required=required)
    assumptions_value = payload["assumptions"]
    if not isinstance(assumptions_value, list):
        raise _error(f"{path}.assumptions", "must be an array")
    assumptions = tuple(
        _string(item, f"{path}.assumptions[{index}]")
        for index, item in enumerate(assumptions_value)
    )
    constants_payload = _object(payload["constants"], f"{path}.constants")
    constants: dict[str, float | str] = {}
    for name, constant in constants_payload.items():
        key = _string(name, f"{path}.constants.<key>")
        if isinstance(constant, str):
            constants[key] = _string(constant, f"{path}.constants.{key}")
        else:
            constants[key] = _finite_number(constant, f"{path}.constants.{key}")
    return BoundEvidence(
        bound_type=_enum(BoundType, payload["bound_type"], f"{path}.bound_type"),
        quantity=_string(payload["quantity"], f"{path}.quantity"),
        upper_bound=_finite_number(payload["upper_bound"], f"{path}.upper_bound", nonnegative=True),
        norm=_string(payload["norm"], f"{path}.norm"),
        scope=_string(payload["scope"], f"{path}.scope"),
        assumptions=assumptions,
        constants=constants,
    )


def evidence_event_from_dict(value: object, path: str = "$") -> EvidenceEvent:
    """Load one evidence event from its version-1 JSON representation."""

    payload = _object(value, path)
    required = {
        "obligation_id",
        "checker",
        "kind",
        "outcome",
        "level",
        "detail",
        "witness",
        "bound",
    }
    _fields(payload, path, required=required)
    level = (
        None
        if payload["level"] is None
        else _enum(EvidenceLevel, payload["level"], f"{path}.level")
    )
    witness = (
        None
        if payload["witness"] is None
        else _witness_from_dict(payload["witness"], f"{path}.witness")
    )
    bound = (
        None if payload["bound"] is None else _bound_from_dict(payload["bound"], f"{path}.bound")
    )
    try:
        return EvidenceEvent(
            obligation_id=_string(payload["obligation_id"], f"{path}.obligation_id"),
            checker=_string(payload["checker"], f"{path}.checker"),
            kind=_enum(EvidenceKind, payload["kind"], f"{path}.kind"),
            outcome=_enum(EvidenceOutcome, payload["outcome"], f"{path}.outcome"),
            level=level,
            detail=_string(payload["detail"], f"{path}.detail"),
            witness=witness,
            bound=bound,
        )
    except ValueError as error:
        raise _error(path, str(error)) from error


def report_from_dict(value: object) -> Report:
    """Validate and load one version-1 decision report."""

    payload = _object(value, "$")
    required = {
        "report_version",
        "aggregation_policy_version",
        "status",
        "decision_evidence",
        "exact_checks",
        "incomplete_reasons",
        "witness",
        "max_sampled_residual",
        "evidence_events",
    }
    _fields(payload, "$", required=required)
    report_version = payload["report_version"]
    if isinstance(report_version, bool) or report_version != REPORT_VERSION:
        raise _error(
            "$.report_version",
            f"unsupported version {report_version!r}; expected {REPORT_VERSION}",
        )
    policy_version = payload["aggregation_policy_version"]
    if isinstance(policy_version, bool) or policy_version != AGGREGATION_POLICY_VERSION:
        raise _error(
            "$.aggregation_policy_version",
            "unsupported aggregation policy "
            f"{policy_version!r}; expected {AGGREGATION_POLICY_VERSION}",
        )
    status = _enum(Status, payload["status"], "$.status")
    decision_evidence = (
        None
        if payload["decision_evidence"] is None
        else _enum(EvidenceLevel, payload["decision_evidence"], "$.decision_evidence")
    )
    if status is Status.INCONCLUSIVE and decision_evidence is not None:
        raise _error("$.decision_evidence", "must be null for an inconclusive report")
    if status is Status.PROVED and decision_evidence not in {
        EvidenceLevel.EXACT,
        EvidenceLevel.RIGOROUS_BOUND,
    }:
        raise _error("$.decision_evidence", "proved reports require exact or rigorous evidence")
    if status is Status.REFUTED and decision_evidence not in {
        EvidenceLevel.EXACT,
        EvidenceLevel.EMPIRICAL,
    }:
        raise _error(
            "$.decision_evidence",
            "version 1 refuted reports require exact or empirical evidence",
        )

    witness = (
        None if payload["witness"] is None else _witness_from_dict(payload["witness"], "$.witness")
    )
    if status is Status.REFUTED and witness is None:
        raise _error("$.witness", "refuted reports require a witness")
    if status is not Status.REFUTED and witness is not None:
        raise _error("$.witness", "only refuted reports may carry a decision witness")

    events_value = payload["evidence_events"]
    if not isinstance(events_value, list):
        raise _error("$.evidence_events", "must be an array")
    events = [
        evidence_event_from_dict(item, f"$.evidence_events[{index}]")
        for index, item in enumerate(events_value)
    ]
    discharged_levels: dict[str, EvidenceLevel] = {}
    for event in events:
        if event.outcome is not EvidenceOutcome.DISCHARGED or event.level is None:
            continue
        previous = discharged_levels.get(event.obligation_id)
        if previous is None or (
            previous is EvidenceLevel.RIGOROUS_BOUND and event.level is EvidenceLevel.EXACT
        ):
            discharged_levels[event.obligation_id] = event.level
    if status is Status.PROVED:
        if not discharged_levels:
            raise _error("$.evidence_events", "proved reports require discharged evidence")
        aggregated_level = (
            EvidenceLevel.EXACT
            if all(level is EvidenceLevel.EXACT for level in discharged_levels.values())
            else EvidenceLevel.RIGOROUS_BOUND
        )
        if decision_evidence is not aggregated_level:
            raise _error(
                "$.decision_evidence",
                "does not match the strongest per-obligation discharged evidence",
            )
    if status is Status.REFUTED and not any(
        event.outcome is EvidenceOutcome.REFUTED
        and event.witness is not None
        and event.witness.to_dict() == witness.to_dict()
        and event.level is decision_evidence
        for event in events
    ):
        raise _error("$.evidence_events", "refuted reports must bind their decision witness")

    sampled_value = payload["max_sampled_residual"]
    if sampled_value == "infinity":
        max_sampled_residual = float("inf")
    else:
        max_sampled_residual = _finite_number(
            sampled_value, "$.max_sampled_residual", nonnegative=True
        )

    return Report(
        status=status,
        decision_evidence=decision_evidence,
        exact_checks=_string_map(payload["exact_checks"], "$.exact_checks"),
        incomplete_reasons=_string_map(payload["incomplete_reasons"], "$.incomplete_reasons"),
        witness=witness,
        max_sampled_residual=max_sampled_residual,
        evidence_events=events,
    )


def dump_report(report: Report, path: str | Path) -> None:
    """Write one deterministic strict-JSON report."""

    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def load_report(path: str | Path) -> Report:
    """Load a strict-JSON report from disk."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise ReportSchemaError(f"{source}: cannot read report: {error}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise ReportSchemaError(f"{source}: invalid JSON: {error}") from error
    return report_from_dict(payload)
