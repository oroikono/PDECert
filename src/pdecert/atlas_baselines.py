"""Reproducible baseline adapters for typed Atlas records.

Baseline outcomes are method-specific diagnostics.  In particular, a finite
grid pass is not a PDECert proof and is never converted into ``Status.PROVED``.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as distribution_version
from itertools import product
from pathlib import Path
from typing import Protocol, runtime_checkable

import mpmath
import sympy

from .corpus import (
    CROSS_ARTIFACT_ATLAS_VERSION,
    CorpusError,
    cross_artifact_atlas_sha256,
    load_cross_artifact_atlas,
)
from .templates import TemplateError, bind_symbolic_candidate, template_from_dict


ATLAS_BASELINE_REPORT_VERSION = 1
FIXED_COLLOCATION_BASELINE_VERSION = 1
FIXED_COLLOCATION_MAX_EVALUATIONS = 1_000_000


class AtlasBaselineError(ValueError):
    """Raised when a typed Atlas baseline run cannot be completed."""


class BaselineOutcome(str, Enum):
    """Method-local outcomes that carry no PDECert proof semantics."""

    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BaselineWitness:
    """A sampled obligation evaluation that can be replayed."""

    constraint: str
    constraint_source: str
    sampled_inputs: Mapping[str, int | float]
    absolute_residual: float | str

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, str) or not self.constraint.strip():
            raise ValueError("witness constraint must be a non-empty string")
        if not isinstance(self.constraint_source, str) or not self.constraint_source.strip():
            raise ValueError("witness constraint source must be a non-empty string")
        if not isinstance(self.sampled_inputs, Mapping):
            raise ValueError("witness sampled inputs must be an object")
        if any(
            not isinstance(name, str) or not name or not _is_finite_number(value)
            for name, value in self.sampled_inputs.items()
        ):
            raise ValueError("sampled inputs must be finite numbers with non-empty names")
        if self.absolute_residual != "infinity":
            if not _is_nonnegative_finite_number(self.absolute_residual):
                raise ValueError("witness residual must be nonnegative and finite or 'infinity'")

    def to_dict(self) -> dict[str, object]:
        """Return a strict-JSON-compatible witness."""

        return {
            "absolute_residual": self.absolute_residual,
            "constraint": self.constraint,
            "constraint_source": self.constraint_source,
            "sampled_inputs": dict(sorted(self.sampled_inputs.items())),
        }


@dataclass(frozen=True)
class BaselineResult:
    """One method-specific result with enforced evidence consistency."""

    outcome: BaselineOutcome
    evidence_kind: str
    evidence_level: str | None
    evaluations: int | None = None
    max_absolute_residual: float | str | None = None
    witness: BaselineWitness | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, BaselineOutcome):
            raise ValueError("outcome must be a BaselineOutcome")
        if self.outcome is BaselineOutcome.PASS:
            expected = ("EMPIRICAL_PASS", "EMPIRICAL")
            if (self.evidence_kind, self.evidence_level) != expected:
                raise ValueError("a baseline pass must carry empirical-pass evidence")
            if (
                self.evaluations is None
                or self.max_absolute_residual is None
                or self.witness is not None
                or self.reason
            ):
                raise ValueError(
                    "a baseline pass requires a residual and evaluation count, with no witness"
                )
        elif self.outcome is BaselineOutcome.FAIL:
            expected = ("NUMERICAL_THRESHOLD_EXCEEDANCE", "EMPIRICAL")
            if (self.evidence_kind, self.evidence_level) != expected:
                raise ValueError("a baseline failure must carry numerical-threshold evidence")
            if (
                self.evaluations is None
                or self.max_absolute_residual is None
                or self.witness is None
                or self.reason
            ):
                raise ValueError("a baseline failure requires a residual, count, and witness")
        else:
            if (self.evidence_kind, self.evidence_level) != ("ABSTENTION", None):
                raise ValueError("an unsupported result must carry abstention evidence")
            if (
                self.evaluations is not None
                or self.max_absolute_residual is not None
                or self.witness is not None
            ):
                raise ValueError("an unsupported result cannot carry evaluations or a witness")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("an unsupported result requires a reason")

        residual = self.max_absolute_residual
        if residual is not None and residual != "infinity":
            if not _is_nonnegative_finite_number(residual):
                raise ValueError("maximum residual must be nonnegative and finite or 'infinity'")
        if self.evaluations is not None and (
            isinstance(self.evaluations, bool)
            or not isinstance(self.evaluations, int)
            or self.evaluations < 1
        ):
            raise ValueError("evaluations must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return a strict-JSON-compatible baseline result."""

        return {
            "evidence_kind": self.evidence_kind,
            "evidence_level": self.evidence_level,
            "evaluations": self.evaluations,
            "max_absolute_residual": self.max_absolute_residual,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "witness": self.witness.to_dict() if self.witness is not None else None,
        }


@runtime_checkable
class AtlasBaselineAdapter(Protocol):
    """Explicit extension boundary for one reproducible Atlas baseline."""

    adapter_id: str
    adapter_version: int
    accepted_artifact_types: tuple[str, ...]
    accepted_solution_semantics: tuple[str, ...]

    def configuration(self) -> Mapping[str, object]:
        """Return strict-JSON-compatible method settings."""

    def evaluate_record(self, record: Mapping[str, object]) -> BaselineResult:
        """Evaluate one already validated Atlas v2 record."""


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) >= 0


def _uniform_axis(
    lower: float,
    upper: float,
    count: int,
    assumptions: frozenset[str],
) -> tuple[int | float, ...]:
    if "integer" in assumptions:
        first = math.ceil(lower)
        last = math.floor(upper)
        available = last - first + 1
        if available <= count:
            return tuple(range(first, last + 1))
        denominator = count - 1
        indices = tuple(
            (index * (available - 1) + denominator // 2) // denominator for index in range(count)
        )
        return tuple(first + index for index in dict.fromkeys(indices))
    denominator = count - 1
    values = [lower]
    for index in range(1, denominator):
        fraction = index / denominator
        value = math.fsum(((1.0 - fraction) * lower, fraction * upper))
        values.append(min(upper, max(lower, value)))
    values.append(upper)
    if any(not math.isfinite(value) or not lower <= value <= upper for value in values):
        raise AtlasBaselineError("could not construct a finite in-domain collocation axis")
    return tuple(values)


def _axis_cardinality(
    lower: float,
    upper: float,
    count: int,
    assumptions: frozenset[str],
) -> int:
    if "integer" in assumptions:
        return min(math.floor(upper) - math.ceil(lower) + 1, count)
    return count


def _absolute_residual(function: Callable[..., object], values: tuple[int | float, ...]) -> float:
    try:
        evaluated = function(*values)
        residual = float(abs(evaluated))
    except (ArithmeticError, TypeError, ValueError):
        return float("inf")
    if not math.isfinite(residual):
        return float("inf")
    return residual


@dataclass(frozen=True)
class FixedCollocationBaseline:
    """Uniform full-condition collocation for symbolic classical solutions."""

    decimal_precision: int = 30
    points_per_axis: int = 5
    tolerance: float = 1e-9

    adapter_id = "fixed_collocation"
    adapter_version = FIXED_COLLOCATION_BASELINE_VERSION
    accepted_artifact_types = ("symbolic_expression",)
    accepted_solution_semantics = ("classical_strong",)

    def __post_init__(self) -> None:
        if (
            isinstance(self.decimal_precision, bool)
            or not isinstance(self.decimal_precision, int)
            or not 15 <= self.decimal_precision <= 100
        ):
            raise ValueError("decimal_precision must be an integer from 15 through 100")
        if (
            isinstance(self.points_per_axis, bool)
            or not isinstance(self.points_per_axis, int)
            or self.points_per_axis < 2
        ):
            raise ValueError("points_per_axis must be an integer of at least two")
        if not _is_finite_number(self.tolerance) or float(self.tolerance) <= 0:
            raise ValueError("tolerance must be finite and positive")
        object.__setattr__(self, "tolerance", float(self.tolerance))

    def configuration(self) -> Mapping[str, object]:
        """Return the complete deterministic grid configuration."""

        return {
            "decimal_precision": self.decimal_precision,
            "include_conditions": True,
            "max_evaluations": FIXED_COLLOCATION_MAX_EVALUATIONS,
            "points_per_axis": self.points_per_axis,
            "sampling": "uniform_tensor_grid_including_endpoints",
            "tolerance": self.tolerance,
        }

    def evaluate_record(self, record: Mapping[str, object]) -> BaselineResult:
        """Evaluate one typed record without assigning a PDECert status."""

        artifact_type = record.get("artifact_type")
        if artifact_type not in self.accepted_artifact_types:
            return BaselineResult(
                outcome=BaselineOutcome.UNSUPPORTED,
                evidence_kind="ABSTENTION",
                evidence_level=None,
                reason=(
                    f"{self.adapter_id} accepts only symbolic_expression artifacts; "
                    f"received {artifact_type!r}"
                ),
            )
        try:
            template = template_from_dict(record["template"])
        except (KeyError, TemplateError, TypeError) as error:
            raise AtlasBaselineError(f"record {record.get('id')!r}: {error}") from error
        if template.solution_semantics not in self.accepted_solution_semantics:
            return BaselineResult(
                outcome=BaselineOutcome.UNSUPPORTED,
                evidence_kind="ABSTENTION",
                evidence_level=None,
                reason=(
                    f"{self.adapter_id} does not support solution semantics "
                    f"{template.solution_semantics!r}"
                ),
            )

        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("fields"), Mapping):
            raise AtlasBaselineError(f"record {record.get('id')!r}: invalid symbolic artifact")
        try:
            case = bind_symbolic_candidate(template, artifact["fields"])
        except TemplateError as error:
            raise AtlasBaselineError(f"record {record.get('id')!r}: {error}") from error

        problem = case.problem
        constraints = problem.pde_residuals + problem.conditions
        variable_sets = tuple(
            tuple(
                variable
                for variable in problem.variables
                if variable in constraint.residual.free_symbols
            )
            for constraint in constraints
        )
        used_variables = {variable for variables in variable_sets for variable in variables}
        cardinalities = {
            variable: _axis_cardinality(
                *problem.domains[variable],
                self.points_per_axis,
                problem.parameter_assumptions.get(variable, frozenset()),
            )
            for variable in used_variables
        }
        total_evaluations = sum(
            math.prod(cardinalities[variable] for variable in variables)
            for variables in variable_sets
        )
        if total_evaluations > FIXED_COLLOCATION_MAX_EVALUATIONS:
            raise AtlasBaselineError(
                f"record {record.get('id')!r}: fixed collocation requires "
                f"{total_evaluations} evaluations, exceeding the "
                f"{FIXED_COLLOCATION_MAX_EVALUATIONS}-evaluation limit"
            )
        axes = {
            variable: _uniform_axis(
                *problem.domains[variable],
                self.points_per_axis,
                problem.parameter_assumptions.get(variable, frozenset()),
            )
            for variable in used_variables
        }
        max_residual = 0.0
        max_witness: BaselineWitness | None = None
        observed_evaluations = 0
        with mpmath.workdps(self.decimal_precision):
            for constraint, variables in zip(constraints, variable_sets, strict=True):
                try:
                    function = sympy.lambdify(
                        variables,
                        constraint.residual,
                        modules="mpmath",
                    )
                except (KeyError, NameError, NotImplementedError, TypeError, ValueError) as error:
                    return BaselineResult(
                        outcome=BaselineOutcome.UNSUPPORTED,
                        evidence_kind="ABSTENTION",
                        evidence_level=None,
                        reason=f"could not compile constraint {constraint.name!r}: {error}",
                    )
                for values in product(*(axes[variable] for variable in variables)):
                    observed_evaluations += 1
                    residual = _absolute_residual(function, values)
                    if residual > max_residual:
                        max_residual = residual
                        rendered_residual: float | str = (
                            residual if math.isfinite(residual) else "infinity"
                        )
                        max_witness = BaselineWitness(
                            constraint=constraint.name,
                            constraint_source=constraint.source or sympy.sstr(constraint.residual),
                            sampled_inputs={
                                str(variable): value
                                for variable, value in zip(variables, values, strict=True)
                            },
                            absolute_residual=rendered_residual,
                        )

        rendered_max: float | str = max_residual if math.isfinite(max_residual) else "infinity"
        if max_residual > self.tolerance:
            if max_witness is None:  # pragma: no cover - protected by the strict comparison above
                raise AtlasBaselineError("failed collocation run did not produce a witness")
            return BaselineResult(
                outcome=BaselineOutcome.FAIL,
                evidence_kind="NUMERICAL_THRESHOLD_EXCEEDANCE",
                evidence_level="EMPIRICAL",
                evaluations=observed_evaluations,
                max_absolute_residual=rendered_max,
                witness=max_witness,
            )
        return BaselineResult(
            outcome=BaselineOutcome.PASS,
            evidence_kind="EMPIRICAL_PASS",
            evidence_level="EMPIRICAL",
            evaluations=observed_evaluations,
            max_absolute_residual=rendered_max,
        )


def _select_records(
    records: Sequence[Mapping[str, object]],
    requested: Sequence[str] | None,
) -> list[Mapping[str, object]]:
    if requested is None:
        return list(records)
    if isinstance(requested, (str, bytes)):
        raise AtlasBaselineError("record_ids must be a sequence of record identifiers")
    identifiers = list(requested)
    if not identifiers:
        raise AtlasBaselineError("record_ids must not be empty when provided")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise AtlasBaselineError("record_ids must contain non-empty strings")
    duplicates = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicates:
        raise AtlasBaselineError("duplicate record id(s): " + ", ".join(duplicates))
    by_id = {str(record["id"]): record for record in records}
    unknown = sorted(set(identifiers) - set(by_id))
    if unknown:
        raise AtlasBaselineError("unknown Atlas record id(s): " + ", ".join(unknown))
    return [by_id[identifier] for identifier in identifiers]


def _package_version() -> str:
    try:
        return distribution_version("pdecert")
    except PackageNotFoundError:
        return "0+unknown"


def _adapter_metadata(adapter: AtlasBaselineAdapter) -> dict[str, object]:
    adapter_id = adapter.adapter_id
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise AtlasBaselineError("baseline adapter id must be a non-empty string")
    version = adapter.adapter_version
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AtlasBaselineError("baseline adapter version must be a positive integer")
    artifact_types = adapter.accepted_artifact_types
    semantics = adapter.accepted_solution_semantics
    for name, values in (
        ("accepted_artifact_types", artifact_types),
        ("accepted_solution_semantics", semantics),
    ):
        if (
            not isinstance(values, tuple)
            or not values
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise AtlasBaselineError(f"baseline adapter {name} must be unique non-empty strings")
    configuration = adapter.configuration()
    if not isinstance(configuration, Mapping) or not all(
        isinstance(key, str) for key in configuration
    ):
        raise AtlasBaselineError("baseline adapter configuration must be an object")
    try:
        normalized_configuration = json.loads(
            json.dumps(configuration, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise AtlasBaselineError(
            f"baseline adapter configuration is not strict JSON: {error}"
        ) from error
    return {
        "accepted_artifact_types": list(artifact_types),
        "accepted_solution_semantics": list(semantics),
        "configuration": normalized_configuration,
        "id": adapter_id,
        "version": version,
    }


def evaluate_atlas_baseline(
    path: str | Path,
    adapter: AtlasBaselineAdapter,
    *,
    record_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run one explicit baseline over selected Atlas v2 records."""

    if not isinstance(adapter, AtlasBaselineAdapter):
        raise TypeError("adapter must implement AtlasBaselineAdapter")
    adapter_metadata = _adapter_metadata(adapter)
    try:
        atlas = load_cross_artifact_atlas(path)
    except (OSError, CorpusError) as error:
        raise AtlasBaselineError(str(error)) from error
    if atlas["atlas_version"] != CROSS_ARTIFACT_ATLAS_VERSION:
        raise AtlasBaselineError(f"expected Atlas version {CROSS_ARTIFACT_ATLAS_VERSION}")
    records = _select_records(atlas["records"], record_ids)
    if not records:
        raise AtlasBaselineError("Atlas contains no records to evaluate")

    evaluations: list[dict[str, object]] = []
    for record in records:
        result = adapter.evaluate_record(record)
        if not isinstance(result, BaselineResult):
            raise AtlasBaselineError("baseline adapter returned an invalid result object")
        artifact = record["artifact"]
        if not isinstance(artifact, Mapping):  # already guaranteed by Atlas validation
            raise AtlasBaselineError(f"record {record['id']!r}: artifact must be an object")
        evaluations.append(
            {
                "artifact_id": artifact["artifact_id"],
                "artifact_type": record["artifact_type"],
                "problem_id": record["problem_id"],
                "record_id": record["id"],
                **result.to_dict(),
            }
        )

    return {
        "adapter": adapter_metadata,
        "atlas": {
            "atlas_version": atlas["atlas_version"],
            "name": atlas["name"],
            "sha256": cross_artifact_atlas_sha256(atlas),
        },
        "baseline_report_version": ATLAS_BASELINE_REPORT_VERSION,
        "evidence_policy": "method_specific_empirical_diagnostics_no_proof",
        "records": evaluations,
        "runtime": {
            "mpmath_version": mpmath.__version__,
            "pdecert_version": _package_version(),
            "python_version": platform.python_version(),
            "sympy_version": sympy.__version__,
        },
    }
