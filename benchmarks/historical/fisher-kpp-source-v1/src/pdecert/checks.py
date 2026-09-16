"""Composable verification checks and their execution registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Protocol

import sympy as sp

from . import core as _core
from .core import Problem, Report, Status
from .evidence import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceLevel,
    EvidenceOutcome,
    Witness,
)


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

    Passing a finite sample must not appear in ``proved_obligations``. Every
    proof or witness must declare the strength of the evidence supporting it.
    """

    proved_obligations: frozenset[str] = frozenset()
    proof_level: EvidenceLevel | None = None
    exact_checks: Mapping[str, str] = field(default_factory=dict)
    incomplete_reasons: Mapping[str, str] = field(default_factory=dict)
    witness: Witness | None = None
    witness_level: EvidenceLevel | None = None
    max_sampled_residual: float = 0.0
    evidence_events: tuple[EvidenceEvent, ...] = ()


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
        evidence: list[EvidenceEvent] = []
        for field_name, expression in context.candidate_fields:
            budget_reason = context.operation_budget_reason(expression)
            for variable in context.problem.variables:
                lower, upper = context.problem.domains[variable]
                check_name = f"{field_name} domain in {variable}"
                obligation_id = context.domain_obligation(field_name, variable)
                if budget_reason is not None:
                    incomplete[check_name] = budget_reason
                    evidence.append(
                        EvidenceEvent(
                            obligation_id=obligation_id,
                            checker=self.name,
                            kind=EvidenceKind.ABSTENTION,
                            outcome=EvidenceOutcome.ABSTAINED,
                            level=None,
                            detail=budget_reason,
                        )
                    )
                    continue
                location, error = _core._run_bounded(
                    lambda: _core._domain_singularity(expression, variable, lower, upper),
                    context.symbolic_timeout,
                )
                if error is not None:
                    incomplete[check_name] = error
                    evidence.append(
                        EvidenceEvent(
                            obligation_id=obligation_id,
                            checker=self.name,
                            kind=EvidenceKind.ABSTENTION,
                            outcome=EvidenceOutcome.ABSTAINED,
                            level=None,
                            detail=error,
                        )
                    )
                    continue
                if location is not None:
                    point, numeric = location
                    witness = Witness(
                        constraint=f"{field_name} domain",
                        point={str(variable): numeric},
                        residual="undefined",
                        reason=f"{field_name} has a singularity at {variable}={sp.sstr(point)}",
                    )
                    evidence.append(
                        EvidenceEvent(
                            obligation_id=obligation_id,
                            checker=self.name,
                            kind=EvidenceKind.EXACT_CERTIFICATE,
                            outcome=EvidenceOutcome.REFUTED,
                            level=EvidenceLevel.EXACT,
                            detail="symbolic singularity analysis located a point in the domain",
                            witness=witness,
                        )
                    )
                    return CheckResult(
                        proved_obligations=frozenset(proved),
                        proof_level=EvidenceLevel.EXACT if proved else None,
                        incomplete_reasons=incomplete,
                        witness=witness,
                        witness_level=EvidenceLevel.EXACT,
                        evidence_events=tuple(evidence),
                    )
                proved.add(obligation_id)
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.EXACT_CERTIFICATE,
                        outcome=EvidenceOutcome.DISCHARGED,
                        level=EvidenceLevel.EXACT,
                        detail="symbolic singularity analysis found no enumerated point in the domain",
                    )
                )
        return CheckResult(
            proved_obligations=frozenset(proved),
            proof_level=EvidenceLevel.EXACT if proved else None,
            incomplete_reasons=incomplete,
            evidence_events=tuple(evidence),
        )


class ExactIdentityChecker:
    """Attempt exact symbolic discharge of residual and condition obligations."""

    name = "exact_identity"

    def check(self, context: CheckContext) -> CheckResult:
        proved: set[str] = set()
        exact: dict[str, str] = {}
        incomplete: dict[str, str] = {}
        evidence: list[EvidenceEvent] = []
        for index, constraint in enumerate(context.constraints):
            obligation_id = context.constraint_obligation(index)
            if reason := context.operation_budget_reason(constraint.residual):
                exact[constraint.name] = "unknown"
                incomplete[constraint.name] = reason
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.ABSTENTION,
                        outcome=EvidenceOutcome.ABSTAINED,
                        level=None,
                        detail=reason,
                    )
                )
                continue
            decision, error = _core._run_bounded(
                lambda: _core._is_zero(constraint.residual),
                context.symbolic_timeout,
            )
            exact[constraint.name] = (
                "identity" if decision is True else "nonzero" if decision is False else "unknown"
            )
            if decision is True:
                proved.add(obligation_id)
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.EXACT_CERTIFICATE,
                        outcome=EvidenceOutcome.DISCHARGED,
                        level=EvidenceLevel.EXACT,
                        detail="symbolic zero-equivalence check established an identity",
                    )
                )
            if error is not None:
                incomplete[constraint.name] = error
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.ABSTENTION,
                        outcome=EvidenceOutcome.ABSTAINED,
                        level=None,
                        detail=error,
                    )
                )
            elif decision is None:
                reason = "symbolic simplification did not decide"
                incomplete[constraint.name] = reason
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.ABSTENTION,
                        outcome=EvidenceOutcome.ABSTAINED,
                        level=None,
                        detail=reason,
                    )
                )
            elif decision is False:
                reason = (
                    "symbolic check found a nonzero constant but this stage did not emit "
                    "a replayable witness"
                )
                incomplete[constraint.name] = reason
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.ABSTENTION,
                        outcome=EvidenceOutcome.ABSTAINED,
                        level=None,
                        detail=reason,
                    )
                )
        return CheckResult(
            proved_obligations=frozenset(proved),
            proof_level=EvidenceLevel.EXACT if proved else None,
            exact_checks=exact,
            incomplete_reasons=incomplete,
            evidence_events=tuple(evidence),
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
        evidence: list[EvidenceEvent] = []
        for constraint_index, constraint in enumerate(context.constraints):
            obligation_id = context.constraint_obligation(constraint_index)
            constraint_max = 0.0
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
                constraint_max = max(constraint_max, residual)
                if residual > context.tolerance:
                    witness = Witness(
                        constraint=constraint.name,
                        point={
                            str(variable): float(value)
                            for variable, value in zip(context.problem.variables, values)
                            if variable in constraint.residual.free_symbols
                        },
                        residual=residual,
                        reason="off-grid evaluation found a violated obligation",
                    )
                    evidence.append(
                        EvidenceEvent(
                            obligation_id=obligation_id,
                            checker=self.name,
                            kind=EvidenceKind.EMPIRICAL_COUNTEREXAMPLE,
                            outcome=EvidenceOutcome.REFUTED,
                            level=EvidenceLevel.EMPIRICAL,
                            detail=(
                                f"deterministic sampling exceeded tolerance {context.tolerance:g}"
                            ),
                            witness=witness,
                        )
                    )
                    return CheckResult(
                        witness=witness,
                        witness_level=EvidenceLevel.EMPIRICAL,
                        max_sampled_residual=max_residual,
                        evidence_events=tuple(evidence),
                    )
            evidence.append(
                EvidenceEvent(
                    obligation_id=obligation_id,
                    checker=self.name,
                    kind=EvidenceKind.EMPIRICAL_PASS,
                    outcome=EvidenceOutcome.OBSERVED_PASS,
                    level=EvidenceLevel.EMPIRICAL,
                    detail=(
                        "deterministic samples did not exceed tolerance; finite sampling "
                        f"cannot discharge the obligation (maximum {constraint_max:g})"
                    ),
                )
            )
        return CheckResult(max_sampled_residual=max_residual, evidence_events=tuple(evidence))


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
    proof_levels: dict[str, EvidenceLevel] = {}
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
        if result.proved_obligations and result.proof_level not in {
            EvidenceLevel.EXACT,
            EvidenceLevel.RIGOROUS_BOUND,
        }:
            raise CheckerError(
                f"checker {checker.name} must attach exact or rigorous-bound evidence "
                "to proved obligations"
            )
        if not result.proved_obligations and result.proof_level is not None:
            raise CheckerError(
                f"checker {checker.name} returned a proof level without proved obligations"
            )
        if result.witness is not None and result.witness_level is None:
            raise CheckerError(f"checker {checker.name} returned a witness without evidence level")
        if result.witness is None and result.witness_level is not None:
            raise CheckerError(f"checker {checker.name} returned a witness level without a witness")
        for event in result.evidence_events:
            if not isinstance(event, EvidenceEvent):
                raise CheckerError(f"checker {checker.name} returned an invalid evidence event")
            if event.checker != checker.name:
                raise CheckerError(
                    f"checker {checker.name} returned evidence attributed to {event.checker}"
                )
            if event.obligation_id not in obligations:
                raise CheckerError(
                    f"checker {checker.name} returned evidence for unknown obligation: "
                    f"{event.obligation_id}"
                )
            if (
                event.outcome is EvidenceOutcome.DISCHARGED
                and event.obligation_id not in result.proved_obligations
            ):
                raise CheckerError(
                    f"checker {checker.name} discharged {event.obligation_id} without "
                    "declaring it proved"
                )
            if (
                event.outcome is EvidenceOutcome.DISCHARGED
                and event.level is not result.proof_level
            ):
                raise CheckerError(
                    f"checker {checker.name} returned discharged evidence whose level "
                    "does not match its proof level"
                )

        event_backed_proofs = {
            event.obligation_id
            for event in result.evidence_events
            if event.outcome is EvidenceOutcome.DISCHARGED and event.level is result.proof_level
        }
        missing_proof_events = set(result.proved_obligations) - event_backed_proofs
        if result.proof_level is EvidenceLevel.RIGOROUS_BOUND and missing_proof_events:
            names = ", ".join(sorted(missing_proof_events))
            raise CheckerError(
                f"checker {checker.name} must attach structured bound evidence for: {names}"
            )
        synthesized_events: list[EvidenceEvent] = []
        if result.proof_level is EvidenceLevel.EXACT:
            for obligation_id in sorted(missing_proof_events):
                synthesized_events.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=checker.name,
                        kind=EvidenceKind.EXACT_CERTIFICATE,
                        outcome=EvidenceOutcome.DISCHARGED,
                        level=EvidenceLevel.EXACT,
                        detail="legacy checker declared exact proof evidence",
                    )
                )
        refuting_events = tuple(
            event for event in result.evidence_events if event.outcome is EvidenceOutcome.REFUTED
        )
        if refuting_events and result.witness is None:
            raise CheckerError(
                f"checker {checker.name} returned refuting evidence without a decision witness"
            )
        if len(refuting_events) > 1:
            raise CheckerError(
                f"checker {checker.name} returned multiple refuting events in one result"
            )
        if refuting_events and (
            refuting_events[0].witness != result.witness
            or refuting_events[0].level is not result.witness_level
        ):
            raise CheckerError(
                f"checker {checker.name} returned refuting evidence that does not match "
                "its decision witness"
            )
        if result.witness is not None and not refuting_events:
            if result.witness_level is EvidenceLevel.RIGOROUS_BOUND:
                raise CheckerError(
                    f"checker {checker.name} returned a rigorous-bound refutation, which "
                    "report schema version 1 does not represent"
                )
            synthesized_events.append(
                EvidenceEvent(
                    obligation_id=_witness_obligation(context, result.witness),
                    checker=checker.name,
                    kind=(
                        EvidenceKind.EMPIRICAL_COUNTEREXAMPLE
                        if result.witness_level is EvidenceLevel.EMPIRICAL
                        else EvidenceKind.EXACT_CERTIFICATE
                    ),
                    outcome=EvidenceOutcome.REFUTED,
                    level=result.witness_level,
                    detail="legacy checker supplied a replayable refutation witness",
                    witness=result.witness,
                )
            )

        proved.update(result.proved_obligations)
        if result.proof_level is not None:
            for obligation in result.proved_obligations:
                previous = proof_levels.get(obligation)
                if previous is None or (
                    previous is EvidenceLevel.RIGOROUS_BOUND
                    and result.proof_level is EvidenceLevel.EXACT
                ):
                    proof_levels[obligation] = result.proof_level
        report.exact_checks.update(result.exact_checks)
        report.incomplete_reasons.update(result.incomplete_reasons)
        report.max_sampled_residual = max(
            report.max_sampled_residual,
            result.max_sampled_residual,
        )
        report.evidence_events.extend(result.evidence_events)
        report.evidence_events.extend(synthesized_events)
        if result.witness is not None:
            report.status = Status.REFUTED
            report.witness = result.witness
            report.decision_evidence = result.witness_level
            return report
        if obligations <= proved:
            report.status = Status.PROVED
            report.decision_evidence = (
                EvidenceLevel.EXACT
                if all(proof_levels[item] is EvidenceLevel.EXACT for item in obligations)
                else EvidenceLevel.RIGOROUS_BOUND
            )
            return report
    return report


def _witness_obligation(context: CheckContext, witness: Witness) -> str:
    """Map a legacy witness label to one declared obligation conservatively."""

    for index, constraint in enumerate(context.constraints):
        if constraint.name == witness.constraint:
            return context.constraint_obligation(index)
    domain_prefix = witness.constraint.removesuffix(" domain")
    domain_matches = sorted(
        obligation
        for obligation in context.obligations
        if obligation.startswith(f"domain:{domain_prefix}:")
    )
    if len(domain_matches) == 1:
        return domain_matches[0]
    raise CheckerError(
        "legacy witness constraint does not identify exactly one declared obligation: "
        f"{witness.constraint}"
    )
