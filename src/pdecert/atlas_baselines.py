"""Reproducible baseline adapters for typed Atlas records.

Baseline outcomes are method-specific diagnostics.  In particular, a finite
grid pass is not a PDECert proof and is never converted into ``Status.PROVED``.
"""

from __future__ import annotations

import ast
import json
import math
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as distribution_version
from itertools import product
from pathlib import Path
from types import MappingProxyType
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
from .core import _run_bounded


ATLAS_BASELINE_REPORT_VERSION = 1
ATLAS_SYMBOLIC_BASELINE_REPORT_VERSION = 2
DIRECT_SYMPY_BASELINE_VERSION = 1
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
        object.__setattr__(self, "sampled_inputs", MappingProxyType(dict(self.sampled_inputs)))

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
                or self.reason is not None
            ):
                raise ValueError(
                    "a baseline pass requires a residual and evaluation count, with no witness"
                )
            if not _is_nonnegative_finite_number(self.max_absolute_residual):
                raise ValueError("a baseline pass requires a finite nonnegative residual")
        elif self.outcome is BaselineOutcome.FAIL:
            expected = ("NUMERICAL_THRESHOLD_EXCEEDANCE", "EMPIRICAL")
            if (self.evidence_kind, self.evidence_level) != expected:
                raise ValueError("a baseline failure must carry numerical-threshold evidence")
            if (
                self.evaluations is None
                or self.max_absolute_residual is None
                or not isinstance(self.witness, BaselineWitness)
                or self.reason is not None
            ):
                raise ValueError("a baseline failure requires a residual, count, and witness")
            if self.witness.absolute_residual != self.max_absolute_residual:
                raise ValueError("failure witness residual must match the maximum residual")
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


@dataclass(frozen=True)
class SymbolicBaselineCheck:
    """One CAS result about a represented expression, not domain regularity."""

    obligation_id: str
    constraint: str
    constraint_source: str
    outcome: str
    materialized_residual: str | None = None
    simplified_residual: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not re.fullmatch(
            r"(?:pde_residuals|conditions)\[[0-9]+\]", self.obligation_id
        ):
            raise ValueError("symbolic check requires a stable obligation id")
        for value in (self.constraint, self.constraint_source):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("symbolic check requires a constraint name and source")
        for value in (self.materialized_residual, self.simplified_residual):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("symbolic residual must be nonempty text or null")
        if self.outcome not in ("zero", "nonzero", "undecided"):
            raise ValueError("symbolic outcome must be zero, nonzero, or undecided")
        if self.outcome == "undecided":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("undecided symbolic check requires a reason")
        elif (
            self.materialized_residual is None
            or self.simplified_residual is None
            or self.reason is not None
        ):
            raise ValueError("decided symbolic check requires both residuals and no reason")
        if self.outcome == "zero" and self.simplified_residual != "0":
            raise ValueError("a zero check must record simplified residual '0'")
        if self.outcome == "nonzero" and self.simplified_residual == "0":
            raise ValueError("a nonzero check cannot record residual '0'")

    def to_dict(self) -> dict[str, object]:
        kinds = {
            "zero": "CAS_RESIDUAL_ZERO",
            "nonzero": "CAS_CONSTANT_NONZERO",
            "undecided": "ABSTENTION",
        }
        return {
            "obligation_id": self.obligation_id,
            "constraint": self.constraint,
            "constraint_source": self.constraint_source,
            "outcome": self.outcome,
            "materialized_residual": self.materialized_residual,
            "simplified_residual": self.simplified_residual,
            "reason": self.reason,
            "evidence_kind": kinds[self.outcome],
            "evidence_level": None if self.outcome == "undecided" else "EXACT",
        }


@dataclass(frozen=True)
class SymbolicBaselineResult:
    """Ordered expression checks with a derived, method-local outcome."""

    checks: tuple[SymbolicBaselineCheck, ...]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if not checks or not all(isinstance(check, SymbolicBaselineCheck) for check in checks):
            raise ValueError("symbolic result requires at least one typed check")
        if len({check.obligation_id for check in checks}) != len(checks):
            raise ValueError("symbolic result contains duplicate obligation ids")
        object.__setattr__(self, "checks", checks)

    @property
    def outcome(self) -> str:
        if any(check.outcome == "nonzero" for check in self.checks):
            return "nonzero"
        return "zero" if all(check.outcome == "zero" for check in self.checks) else "undecided"

    def to_dict(self) -> dict[str, object]:
        return {
            "result_type": "symbolic_residuals",
            "scope": "represented_residuals_and_conditions_only",
            "outcome": self.outcome,
            "checks": [check.to_dict() for check in self.checks],
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

    def evaluate_record(
        self, record: Mapping[str, object]
    ) -> BaselineResult | SymbolicBaselineResult:
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


@dataclass(frozen=True)
class DirectSympyBaseline:
    """Bounded CAS simplification of every represented residual and condition."""

    symbolic_timeout: float = 2.0
    max_expression_ops: int = 10_000

    adapter_id = "direct_sympy"
    adapter_version = DIRECT_SYMPY_BASELINE_VERSION
    report_version = ATLAS_SYMBOLIC_BASELINE_REPORT_VERSION
    accepted_artifact_types = ("symbolic_expression",)
    accepted_solution_semantics = ("classical_strong",)

    def __post_init__(self) -> None:
        if (
            not _is_finite_number(self.symbolic_timeout)
            or not 0.001 <= self.symbolic_timeout <= 3600
        ):
            raise ValueError("symbolic_timeout must be from 0.001 through 3600 seconds")
        if (
            isinstance(self.max_expression_ops, bool)
            or not isinstance(self.max_expression_ops, int)
            or self.max_expression_ops < 1
        ):
            raise ValueError("max_expression_ops must be a positive integer")
        object.__setattr__(self, "symbolic_timeout", float(self.symbolic_timeout))

    def configuration(self) -> Mapping[str, object]:
        return {
            "pipeline": ["simplify", "cancel", "trigsimp"],
            "include_conditions": True,
            "domain_checks": False,
            "nonzero_policy": "finite_real_constant_only",
            "float_policy": "abstain_on_inexact_literals",
            "symbolic_timeout_seconds": self.symbolic_timeout,
            "max_expression_ops": self.max_expression_ops,
            "deadline_scope": "binding_and_each_obligation_after_atlas_validation",
        }

    def evaluate_record(
        self, record: Mapping[str, object]
    ) -> BaselineResult | SymbolicBaselineResult:
        if record.get("artifact_type") not in self.accepted_artifact_types:
            return BaselineResult(
                BaselineOutcome.UNSUPPORTED,
                "ABSTENTION",
                None,
                reason="direct_sympy accepts only symbolic_expression artifacts",
            )
        raw_template = record["template"]
        if raw_template.get("solution_semantics") not in self.accepted_solution_semantics:
            return BaselineResult(
                BaselineOutcome.UNSUPPORTED,
                "ABSTENTION",
                None,
                reason="direct_sympy accepts only classical_strong solution semantics",
            )
        fields = record["artifact"]["fields"]
        obligations = [
            (f"{group}[{index}]", constraint["name"], constraint["expression"])
            for group in ("pde_residuals", "conditions")
            for index, constraint in enumerate(raw_template[group])
        ]

        def abstain(reason: str) -> SymbolicBaselineResult:
            return SymbolicBaselineResult(
                tuple(
                    SymbolicBaselineCheck(identifier, name, source, "undecided", reason=reason)
                    for identifier, name, source in obligations
                )
            )

        # Eager parsing can cancel Float literals to integer zero, so inspect
        # source syntax before binding. Decimal domain bounds are metadata.
        sources = [*fields.values(), *(source for _, _, source in obligations)]
        if any(
            (isinstance(node, ast.Constant) and isinstance(node.value, float))
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Float"
            )
            for source in sources
            for node in ast.walk(ast.parse(source, mode="eval"))
        ):
            return abstain(
                "inexact numeric literal in field or operator; use exact rational syntax"
            )
        case, error = _run_bounded(
            lambda: bind_symbolic_candidate(template_from_dict(raw_template), fields),
            self.symbolic_timeout,
        )
        if error is not None:
            return abstain(f"candidate binding: {error}")
        constraints = case.problem.pde_residuals + case.problem.conditions
        checks = []
        for (identifier, name, source), constraint in zip(obligations, constraints, strict=True):
            checked, error = _run_bounded(
                lambda expression=constraint.residual: self._check_expression(expression),
                self.symbolic_timeout,
            )
            if error is not None:
                checks.append(
                    SymbolicBaselineCheck(identifier, name, source, "undecided", reason=error)
                )
            else:
                outcome, materialized, simplified, reason = checked
                checks.append(
                    SymbolicBaselineCheck(
                        identifier, name, source, outcome, materialized, simplified, reason
                    )
                )
        return SymbolicBaselineResult(tuple(checks))

    def _check_expression(
        self, expression: sympy.Expr
    ) -> tuple[str, str | None, str | None, str | None]:
        operations = int(sympy.count_ops(expression))
        if operations > self.max_expression_ops:
            return (
                "undecided",
                None,
                None,
                f"input expression has {operations} operations, exceeding {self.max_expression_ops}",
            )
        materialized = sympy.sstr(expression)
        if (
            expression.has(sympy.Float, sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)
            or expression.is_real is False
        ):
            return "undecided", materialized, None, "inexact, non-finite, or non-real residual"
        simplified = sympy.trigsimp(sympy.cancel(sympy.simplify(expression)))
        rendered = sympy.sstr(simplified)
        if (
            simplified.has(sympy.Float, sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)
            or simplified.is_real is False
        ):
            return (
                "undecided",
                materialized,
                rendered,
                "inexact, non-finite, or non-real simplified residual",
            )
        if simplified == 0 or simplified.is_zero is True:
            return "zero", materialized, "0", None
        if (
            not simplified.free_symbols
            and simplified.is_real is True
            and simplified.is_finite is True
            and simplified.is_zero is False
        ):
            return "nonzero", materialized, rendered, None
        return (
            "undecided",
            materialized,
            rendered,
            "CAS did not establish zero or a finite real nonzero constant",
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
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
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
    if adapter_id == "fixed_collocation":
        try:
            expected = FixedCollocationBaseline(
                decimal_precision=normalized_configuration.get("decimal_precision"),
                points_per_axis=normalized_configuration.get("points_per_axis"),
                tolerance=normalized_configuration.get("tolerance"),
            )
        except ValueError as error:
            raise AtlasBaselineError(f"invalid fixed_collocation configuration: {error}") from error
        if (
            version != expected.adapter_version
            or artifact_types != expected.accepted_artifact_types
            or semantics != expected.accepted_solution_semantics
            or json.dumps(normalized_configuration, sort_keys=True)
            != json.dumps(expected.configuration(), sort_keys=True)
        ):
            raise AtlasBaselineError("fixed_collocation metadata must match its versioned contract")
    if adapter_id == "direct_sympy":
        try:
            expected = DirectSympyBaseline(
                symbolic_timeout=normalized_configuration.get("symbolic_timeout_seconds"),
                max_expression_ops=normalized_configuration.get("max_expression_ops"),
            )
        except ValueError as error:
            raise AtlasBaselineError(f"invalid direct_sympy configuration: {error}") from error
        if (
            version != expected.adapter_version
            or artifact_types != expected.accepted_artifact_types
            or semantics != expected.accepted_solution_semantics
            or json.dumps(normalized_configuration, sort_keys=True)
            != json.dumps(expected.configuration(), sort_keys=True)
        ):
            raise AtlasBaselineError("direct_sympy metadata must match its versioned contract")
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
    report_version = getattr(adapter, "report_version", ATLAS_BASELINE_REPORT_VERSION)
    if type(report_version) is not int or report_version not in (1, 2):
        raise AtlasBaselineError("baseline report_version must be 1 or 2")
    reserved_versions = {"fixed_collocation": 1, "direct_sympy": 2}
    if (
        adapter_metadata["id"] in reserved_versions
        and report_version != reserved_versions[adapter_metadata["id"]]
    ):
        raise AtlasBaselineError("adapter report_version does not match its versioned contract")
    try:
        atlas = load_cross_artifact_atlas(path)
    except (OSError, CorpusError) as error:
        raise AtlasBaselineError(str(error)) from error
    if atlas["atlas_version"] != CROSS_ARTIFACT_ATLAS_VERSION:
        raise AtlasBaselineError(f"expected Atlas version {CROSS_ARTIFACT_ATLAS_VERSION}")
    records = _select_records(atlas["records"], record_ids)
    if not records:
        raise AtlasBaselineError("Atlas contains no records to evaluate")
    atlas_digest = cross_artifact_atlas_sha256(atlas)

    evaluations: list[dict[str, object]] = []
    for record in records:
        artifact_type = record["artifact_type"]
        semantics = record["template"]["solution_semantics"]
        if artifact_type not in adapter_metadata["accepted_artifact_types"]:
            result = BaselineResult(
                BaselineOutcome.UNSUPPORTED,
                "ABSTENTION",
                None,
                reason=(
                    f"{adapter_metadata['id']} accepts artifact types "
                    f"{adapter_metadata['accepted_artifact_types']}; received {artifact_type!r}"
                ),
            )
        elif semantics not in adapter_metadata["accepted_solution_semantics"]:
            result = BaselineResult(
                BaselineOutcome.UNSUPPORTED,
                "ABSTENTION",
                None,
                reason=(
                    f"{adapter_metadata['id']} does not support solution semantics {semantics!r}"
                ),
            )
        else:
            # Adapters may keep local working state, but the evaluated input and
            # the source identity attached to the report must remain unchanged.
            working_record = deepcopy(record)
            result = adapter.evaluate_record(working_record)
            try:
                unchanged = json.dumps(
                    working_record, sort_keys=True, allow_nan=False
                ) == json.dumps(record, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError, RecursionError) as error:
                raise AtlasBaselineError(
                    f"baseline adapter modified input record {record['id']!r}: {error}"
                ) from error
            if not unchanged:
                raise AtlasBaselineError(f"baseline adapter modified input record {record['id']!r}")
        if not isinstance(result, (BaselineResult, SymbolicBaselineResult)):
            raise AtlasBaselineError("baseline adapter returned an invalid result object")
        if isinstance(result, SymbolicBaselineResult):
            if report_version != ATLAS_SYMBOLIC_BASELINE_REPORT_VERSION:
                raise AtlasBaselineError("symbolic baseline results require report_version 2")
            expected_checks = [
                (f"{group}[{index}]", constraint["name"], constraint["expression"])
                for group in ("pde_residuals", "conditions")
                for index, constraint in enumerate(record["template"][group])
            ]
            if (
                artifact_type != "symbolic_expression"
                or [
                    (check.obligation_id, check.constraint, check.constraint_source)
                    for check in result.checks
                ]
                != expected_checks
            ):
                raise AtlasBaselineError(
                    "symbolic result must cover every represented obligation in order"
                )
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
            "sha256": atlas_digest,
        },
        "baseline_report_version": report_version,
        "evidence_policy": (
            "method_specific_empirical_diagnostics_no_proof"
            if report_version == 1
            else "method_specific_obligation_diagnostics_no_global_verdict"
        ),
        "records": evaluations,
        "runtime": {
            "mpmath_version": mpmath.__version__,
            "pdecert_version": _package_version(),
            "python_version": platform.python_version(),
            "sympy_version": sympy.__version__,
        },
    }
