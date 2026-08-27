"""Minimal third-party-style checker using the public extension API."""

from __future__ import annotations

import sympy as sp

from pdecert import CheckContext, CheckResult, EvidenceLevel


class ExpandedPolynomialChecker:
    """Prove polynomial identities by checking every expanded coefficient."""

    name = "expanded_polynomial_identity"

    def check(self, context: CheckContext) -> CheckResult:
        proved: set[str] = set()
        exact: dict[str, str] = {}
        for index, constraint in enumerate(context.constraints):
            try:
                polynomial = sp.Poly(
                    sp.expand(constraint.residual),
                    *context.problem.variables,
                )
            except sp.PolynomialError:
                continue
            if polynomial.is_zero:
                proved.add(context.constraint_obligation(index))
                exact[constraint.name] = "identity"
        return CheckResult(
            proved_obligations=frozenset(proved),
            proof_level=EvidenceLevel.EXACT if proved else None,
            exact_checks=exact,
        )
