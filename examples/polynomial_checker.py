"""Minimal third-party-style checker using the public extension API."""

from __future__ import annotations

import sympy as sp

from pdecert import (
    CheckContext,
    CheckResult,
    EvidenceEvent,
    EvidenceKind,
    EvidenceLevel,
    EvidenceOutcome,
)


class ExpandedPolynomialChecker:
    """Prove polynomial identities by checking every expanded coefficient."""

    name = "expanded_polynomial_identity"

    def check(self, context: CheckContext) -> CheckResult:
        proved: set[str] = set()
        exact: dict[str, str] = {}
        evidence: list[EvidenceEvent] = []
        for index, constraint in enumerate(context.constraints):
            try:
                polynomial = sp.Poly(
                    sp.expand(constraint.residual),
                    *context.problem.variables,
                )
            except sp.PolynomialError:
                continue
            if polynomial.is_zero:
                obligation_id = context.constraint_obligation(index)
                proved.add(obligation_id)
                exact[constraint.name] = "identity"
                evidence.append(
                    EvidenceEvent(
                        obligation_id=obligation_id,
                        checker=self.name,
                        kind=EvidenceKind.EXACT_CERTIFICATE,
                        outcome=EvidenceOutcome.DISCHARGED,
                        level=EvidenceLevel.EXACT,
                        detail="expanded polynomial coefficients are identically zero",
                    )
                )
        return CheckResult(
            proved_obligations=frozenset(proved),
            proof_level=EvidenceLevel.EXACT if proved else None,
            exact_checks=exact,
            evidence_events=tuple(evidence),
        )
