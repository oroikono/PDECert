"""Composable verification checks and their execution registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Protocol

import sympy as sp

from . import core as _core
from .core import Problem, Report, Status, Witness


@dataclass(frozen=True)
class CheckContext:
    """Immutable inputs shared by every checker in one verification run."""

    problem: Problem
    candidate_fields: tuple[tuple[str, sp.Expr], ...]
    tolerance: float
    samples_per_axis: int
    symbolic_timeout: float | None
    max_expression_ops: int | None = None

    @property
    def constraints(self):
        """Return all PDE and represented condition obligations in stable order."""

        return self.problem.pde_residuals + self.problem.conditions

    def domain_obligation(self, field_name: str, variable: sp.Symbol) -> str:
        """Return the stable identifier for one field/domain obligation."""

        return f"domain:{field_name}:{variable}"

    def constraint_obligation(self, index: int) -> str:
        """Return the stable identifier for one residual or condition obligation."""

        return f"constraint:{index}"

    @property
    def obligations(self) -> frozenset[str]:
        """Return every obligation that must be proved before acceptance."""

        domains = {
            self.domain_obligation(field_name, variable)
            for field_name, _ in self.candidate_fields
            for variable in self.problem.variables
        }
        constraints = {
            self.constraint_obligation(index) for index, _ in enumerate(self.constraints)
        }
        return frozenset(domains | constraints)

    def operation_budget_reason(self, expression: sp.Expr) -> str | None:
        """Explain why an expression exceeds the configured structural budget."""

        if self.max_expression_ops is None:
            return None
        operations = int(sp.count_ops(expression, visual=False))
        if operations <= self.max_expression_ops:
            return None
        return (
            f"input expression has {operations} operations, exceeding the "
            f"configured limit of {self.max_expression_ops}"
        )


@dataclass(frozen=True)
class CheckResult:
    """Partial evidence returned by one checker.

    Passing a finite sample must not appear in ``proved_obligations``. A checker
    should report a refutation only with a concrete witness.
    """

    proved_obligations: frozenset[str] = frozenset()
    exact_checks: Mapping[str, str] = field(default_factory=dict)
    incomplete_reasons: Mapping[str, str] = field(default_factory=dict)
    witness: Witness | None = None
    max_sampled_residual: float = 0.0


class Checker(Protocol):
    """Public protocol implemented by a verification checker."""

    name: str

    def check(self, context: CheckContext) -> CheckResult:
        """Return partial proof or refutation evidence for ``context``."""


class CheckerError(RuntimeError):
    """Raised when a checker violates the extension contract or fails."""


@dataclass(frozen=True)
class CheckerRegistry:
    """Ordered, immutable collection of uniquely named checkers."""

    checkers: tuple[Checker, ...] = ()

    def __post_init__(self) -> None:
        names: set[str] = set()
        for checker in self.checkers:
            name = getattr(checker, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise ValueError("checker names must be non-empty strings")
            if name in names:
                raise ValueError(f"duplicate checker name: {name}")
            if not callable(getattr(checker, "check", None)):
                raise TypeError(f"checker {name} must define check(context)")
            names.add(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Return checker names in execution order."""

        return tuple(checker.name for checker in self.checkers)

    def with_checker(self, checker: Checker, *, before: str | None = None) -> CheckerRegistry:
        """Return a new registry containing ``checker``.

        By default the checker is appended. ``before`` inserts it before a
        registered checker without mutating the original registry.
        """

        if before is None:
            return CheckerRegistry(self.checkers + (checker,))
        if before not in self.names:
            raise ValueError(f"unknown checker insertion point: {before}")
        index = self.names.index(before)
        return CheckerRegistry(self.checkers[:index] + (checker,) + self.checkers[index:])


class DomainChecker:
    """Prove enumerated domain checks or refute a concrete singularity."""

    name = "domain"

    def check(self, context: CheckContext) -> CheckResult:
        proved: set[str] = set()
        incomplete: dict[str, str] = {}
        for field_name, expression in context.candidate_fields:
            budget_reason = context.operation_budget_reason(expression)
            for variable in context.problem.variables:
                lower, upper = context.problem.domains[variable]
                check_name = f"{field_name} domain in {variable}"
                if budget_reason is not None:
                    incomplete[check_name] = budget_reason
                    continue
                location, error = _core._run_bounded(
                    lambda: _core._domain_singularity(expression, variable, lower, upper),
                    context.symbolic_timeout,
                )
                if error is not None:
                    incomplete[check_name] = error
                    continue
                if location is not None:
                    point, numeric = location
                    return CheckResult(
                        proved_obligations=frozenset(proved),
                        incomplete_reasons=incomplete,
                        witness=Witness(
                            constraint=f"{field_name} domain",
                            point={str(variable): numeric},
                            residual="undefined",
                            reason=(
                                f"{field_name} has a singularity at {variable}={sp.sstr(point)}"
                            ),
                        ),
                    )
                proved.add(context.domain_obligation(field_name, variable))
        return CheckResult(
            proved_obligations=frozenset(proved),
            incomplete_reasons=incomplete,
        )


class ExactIdentityChecker:
    """Attempt exact symbolic discharge of residual and condition obligations."""

    name = "exact_identity"

    def check(self, context: CheckContext) -> CheckResult:
        proved: set[str] = set()
        exact: dict[str, str] = {}
        incomplete: dict[str, str] = {}
        for index, constraint in enumerate(context.constraints):
            if reason := context.operation_budget_reason(constraint.residual):
                exact[constraint.name] = "unknown"
                incomplete[constraint.name] = reason
                continue
            decision, error = _core._run_bounded(
                lambda: _core._is_zero(constraint.residual),
                context.symbolic_timeout,
            )
            exact[constraint.name] = (
                "identity" if decision is True else "nonzero" if decision is False else "unknown"
            )
            if decision is True:
                proved.add(context.constraint_obligation(index))
            if error is not None:
                incomplete[constraint.name] = error
            elif decision is None:
                incomplete[constraint.name] = "symbolic simplification did not decide"
        return CheckResult(
            proved_obligations=frozenset(proved),
            exact_checks=exact,
            incomplete_reasons=incomplete,
        )


class OffGridCounterexampleChecker:
    """Search deterministic off-grid samples for a concrete violation."""

    name = "off_grid_counterexample"

    def check(self, context: CheckContext) -> CheckResult:
        axes = [
            _core._sample_points(context.problem, variable, context.samples_per_axis)
            for variable in context.problem.variables
        ]
        max_residual = 0.0
        for constraint in context.constraints:
            for values in product(*axes):
                try:
                    residual = _core._numeric_value(
                        constraint.residual,
                        context.problem.variables,
                        values,
                    )
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    residual = float("inf")
                max_residual = max(max_residual, residual)
                if residual > context.tolerance:
                    return CheckResult(
                        witness=Witness(
                            constraint=constraint.name,
                            point={
                                str(variable): float(value)
                                for variable, value in zip(context.problem.variables, values)
                                if variable in constraint.residual.free_symbols
                            },
                            residual=residual,
                            reason="off-grid evaluation found a violated obligation",
                        ),
                        max_sampled_residual=max_residual,
                    )
        return CheckResult(max_sampled_residual=max_residual)


def default_checker_registry() -> CheckerRegistry:
    """Return a fresh registry containing the supported built-in pipeline."""

    return CheckerRegistry(
        (
            DomainChecker(),
            ExactIdentityChecker(),
            OffGridCounterexampleChecker(),
        )
    )


def run_checks(context: CheckContext, registry: CheckerRegistry) -> Report:
    """Run ``registry`` and conservatively aggregate its partial evidence."""

    report = Report(status=Status.INCONCLUSIVE)
    proved: set[str] = set()
    obligations = context.obligations
    for checker in registry.checkers:
        try:
            result = checker.check(context)
        except Exception as error:
            raise CheckerError(
                f"checker {checker.name} failed with {type(error).__name__}: {error}"
            ) from error
        if not isinstance(result, CheckResult):
            raise CheckerError(
                f"checker {checker.name} returned {type(result).__name__}, not CheckResult"
            )
        if result.witness is not None and not isinstance(result.witness, Witness):
            raise CheckerError(f"checker {checker.name} returned an invalid witness")
        unknown = set(result.proved_obligations) - obligations
        if unknown:
            names = ", ".join(sorted(unknown))
            raise CheckerError(f"checker {checker.name} proved unknown obligation(s): {names}")
        proved.update(result.proved_obligations)
        report.exact_checks.update(result.exact_checks)
        report.incomplete_reasons.update(result.incomplete_reasons)
        report.max_sampled_residual = max(
            report.max_sampled_residual,
            result.max_sampled_residual,
        )
        if result.witness is not None:
            report.status = Status.REFUTED
            report.witness = result.witness
            return report
        if obligations <= proved:
            report.status = Status.PROVED
            return report
    return report
