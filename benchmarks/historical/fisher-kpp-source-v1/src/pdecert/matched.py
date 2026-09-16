"""Representation-neutral cases evaluated by distinct verification backends."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .artifacts import CallableCandidate, SolutionArtifact, SymbolicCandidate
from .autodiff import AutodiffProblem
from .core import Problem, Report, verify_artifact


@dataclass(frozen=True)
class EvaluationLane:
    """One backend-specific problem and candidate bound to a matched case."""

    name: str
    problem: Problem | AutodiffProblem
    artifact: SolutionArtifact

    def __post_init__(self) -> None:
        _validate_name(self.name, "lane name")
        if isinstance(self.problem, Problem) and isinstance(self.artifact, SymbolicCandidate):
            return
        if isinstance(self.problem, AutodiffProblem) and isinstance(
            self.artifact, CallableCandidate
        ):
            return
        raise TypeError(
            "unsupported problem/artifact pair in evaluation lane: "
            f"{type(self.problem).__name__} and {type(self.artifact).__name__}"
        )


@dataclass(frozen=True)
class MatchedCase:
    """One mathematical case represented by two or more evaluation lanes.

    Constructing a matched case records the maintainer's assertion that each
    lane represents the same mathematical problem. PDECert validates structural
    invariants, but it does not prove equivalence between backend-specific
    problem encodings.
    """

    case_id: str
    coordinate_names: tuple[str, ...]
    field_names: tuple[str, ...]
    solution_semantics: str
    lanes: tuple[EvaluationLane, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))
        object.__setattr__(self, "field_names", tuple(self.field_names))
        object.__setattr__(self, "lanes", tuple(self.lanes))
        _validate_name(self.case_id, "case id")
        _validate_names(self.coordinate_names, "coordinate names")
        _validate_names(self.field_names, "field names")
        _validate_name(self.solution_semantics, "solution semantics")
        if len(self.lanes) < 2:
            raise ValueError("matched cases require at least two evaluation lanes")
        if any(not isinstance(lane, EvaluationLane) for lane in self.lanes):
            raise TypeError("matched case lanes must be EvaluationLane objects")

        lane_names = tuple(lane.name for lane in self.lanes)
        if len(lane_names) != len(set(lane_names)):
            raise ValueError("evaluation lane names must be unique")

        for lane in self.lanes:
            coordinates = _problem_coordinate_names(lane.problem)
            if coordinates != self.coordinate_names:
                raise ValueError(
                    f"lane {lane.name!r} coordinates {coordinates!r} do not match "
                    f"{self.coordinate_names!r}"
                )
            if lane.artifact.field_names != self.field_names:
                raise ValueError(
                    f"lane {lane.name!r} fields {lane.artifact.field_names!r} do not match "
                    f"{self.field_names!r}"
                )


@dataclass(frozen=True)
class LaneVerificationOptions:
    """Backend options for one lane of a matched evaluation."""

    tolerance: float | None = None
    samples_per_axis: int = 5
    symbolic_timeout: float | None = None
    max_expression_ops: int | None = None

    def __post_init__(self) -> None:
        if self.tolerance is not None and (
            not math.isfinite(self.tolerance) or self.tolerance <= 0
        ):
            raise ValueError("tolerance must be finite and positive")
        if (
            isinstance(self.samples_per_axis, bool)
            or not isinstance(self.samples_per_axis, int)
            or self.samples_per_axis < 2
        ):
            raise ValueError("samples_per_axis must be an integer of at least two")
        if self.symbolic_timeout is not None and (
            not math.isfinite(self.symbolic_timeout) or self.symbolic_timeout <= 0
        ):
            raise ValueError("symbolic_timeout must be finite and positive")
        if self.max_expression_ops is not None and (
            isinstance(self.max_expression_ops, bool)
            or not isinstance(self.max_expression_ops, int)
            or self.max_expression_ops < 1
        ):
            raise ValueError("max_expression_ops must be a positive integer")


@dataclass(frozen=True)
class LaneReport:
    """One lane's report with its representation identity retained."""

    name: str
    artifact_kind: str
    report: Report

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible lane report."""

        return {
            "name": self.name,
            "artifact_kind": self.artifact_kind,
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class MatchedReport:
    """Per-lane results for one matched case, intentionally without aggregation."""

    case_id: str
    coordinate_names: tuple[str, ...]
    field_names: tuple[str, ...]
    solution_semantics: str
    lanes: tuple[LaneReport, ...]

    @property
    def reports(self) -> Mapping[str, Report]:
        """Return reports keyed by lane name without discarding lane identity."""

        return MappingProxyType({lane.name: lane.report for lane in self.lanes})

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report without inventing a shared status."""

        return {
            "case_id": self.case_id,
            "coordinate_names": list(self.coordinate_names),
            "field_names": list(self.field_names),
            "solution_semantics": self.solution_semantics,
            "lanes": [lane.to_dict() for lane in self.lanes],
        }


def verify_matched_case(
    case: MatchedCase,
    *,
    options: Mapping[str, LaneVerificationOptions] | None = None,
) -> MatchedReport:
    """Evaluate every lane while preserving backend-specific evidence.

    The function does not combine lane statuses. In particular, an exact
    symbolic proof does not promote a sampled callable result, and an
    inconclusive lane does not weaken a proof produced for a different artifact.
    """

    configured = dict(options or {})
    lane_names = {lane.name for lane in case.lanes}
    unknown = sorted(set(configured) - lane_names)
    if unknown:
        raise ValueError(f"options provided for unknown evaluation lane(s): {', '.join(unknown)}")
    if any(not isinstance(value, LaneVerificationOptions) for value in configured.values()):
        raise TypeError("matched-case options must be LaneVerificationOptions objects")

    results: list[LaneReport] = []
    for lane in case.lanes:
        lane_options = configured.get(lane.name, LaneVerificationOptions())
        if isinstance(lane.artifact, CallableCandidate) and (
            lane_options.symbolic_timeout is not None or lane_options.max_expression_ops is not None
        ):
            raise ValueError(
                f"symbolic resource limits do not apply to callable lane {lane.name!r}"
            )

        if isinstance(lane.artifact, SymbolicCandidate):
            report = verify_artifact(
                lane.problem,
                lane.artifact,
                tolerance=lane_options.tolerance,
                samples_per_axis=lane_options.samples_per_axis,
                symbolic_timeout=lane_options.symbolic_timeout,
                max_expression_ops=lane_options.max_expression_ops,
            )
        else:
            report = verify_artifact(
                lane.problem,
                lane.artifact,
                tolerance=lane_options.tolerance,
                samples_per_axis=lane_options.samples_per_axis,
            )
        results.append(LaneReport(lane.name, lane.artifact.kind, report))

    return MatchedReport(
        case_id=case.case_id,
        coordinate_names=case.coordinate_names,
        field_names=case.field_names,
        solution_semantics=case.solution_semantics,
        lanes=tuple(results),
    )


def _problem_coordinate_names(problem: Problem | AutodiffProblem) -> tuple[str, ...]:
    if isinstance(problem, Problem):
        return tuple(str(variable) for variable in problem.variables)
    return problem.variables


def _validate_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _validate_names(values: tuple[str, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
