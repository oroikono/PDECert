"""Deterministic, offline demonstration of PDECert's evidence contract."""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp

from .agents import AgentEvaluation, AgentProposal, AgentTrace, evaluate_agent_proposal
from .artifacts import SymbolicCandidate
from .core import Status
from .evidence import EvidenceKind
from .schema import VerificationCase, case_from_dict


QUICKSTART_VERSION = 1
_TOLERANCE = 1e-9
_SAMPLES_PER_AXIS = 5
_MAX_EXPRESSION_OPS = 10_000


def _trusted_heat_case(exact: sp.Expr) -> VerificationCase:
    """Build the trusted problem independently of every proposed candidate."""

    return case_from_dict(
        {
            "schema_version": 3,
            "name": "one-dimensional heat equation",
            "variables": ["x", "t"],
            "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            "parameters": {},
            "pde_residuals": [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}],
            "conditions": [
                {
                    "name": "initial condition",
                    "expression": "At(u, t, 0) - sin(pi*x)",
                },
                {"name": "left boundary", "expression": "At(u, x, 0)"},
                {"name": "right boundary", "expression": "At(u, x, 1)"},
            ],
            "fields": {"u": sp.sstr(exact)},
        }
    )


def _proposal(
    proposal_id: str,
    expression: sp.Expr,
    *,
    parent_proposal_id: str | None = None,
) -> AgentProposal:
    return AgentProposal(
        proposal_id=proposal_id,
        generator="offline-recorded-example",
        artifact=SymbolicCandidate.from_expressions({"u": expression}),
        raw_output=sp.sstr(expression),
        parent_proposal_id=parent_proposal_id,
        metadata={"execution": "offline", "role": "demonstration-fixture"},
    )


def _evaluate(case: VerificationCase, proposal: AgentProposal) -> AgentEvaluation:
    return evaluate_agent_proposal(
        case,
        proposal,
        tolerance=_TOLERANCE,
        samples_per_axis=_SAMPLES_PER_AXIS,
        symbolic_timeout=None,
        max_expression_ops=_MAX_EXPRESSION_OPS,
    )


def _scenario(
    scenario_id: str,
    lesson: str,
    expected_status: Status,
    report: Mapping[str, object],
) -> dict[str, object]:
    observed_status = report["status"]
    return {
        "id": scenario_id,
        "lesson": lesson,
        "expected_status": expected_status.value,
        "observed_status": observed_status,
        "passed": observed_status == expected_status.value,
        "report": dict(report),
    }


def run_quickstart() -> dict[str, object]:
    """Run the bundled offline demonstration and return strict-JSON-safe data.

    The recorded proposals are deterministic fixtures; no model or remote
    service is contacted. Exact symbolic evidence may prove the represented
    obligations. Empirical checks may refute or abstain, but never prove.
    """

    x, t = sp.symbols("x t", real=True)
    exact = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    boundary_defect = exact + x / 10
    below_tolerance = exact + sp.Rational(1, 10**14) * t * x * (1 - x)
    trusted_case = _trusted_heat_case(exact)

    rejected = _evaluate(trusted_case, _proposal("attempt-1", boundary_defect))
    repaired = _evaluate(
        trusted_case,
        _proposal("attempt-2", exact, parent_proposal_id="attempt-1"),
    )
    abstained = _evaluate(trusted_case, _proposal("tolerance-probe", below_tolerance))
    trace = AgentTrace(
        run_id="offline-heat-repair-01",
        problem_id="heat-classical-01",
        evaluations=(rejected, repaired),
    )

    scenarios = [
        _scenario(
            "exact-symbolic-proof",
            "Exact identities discharge the PDE, initial condition, and boundaries.",
            Status.PROVED,
            repaired.report.to_dict(),
        ),
        _scenario(
            "boundary-counterexample",
            "A PDE-satisfying expression is rejected by a concrete boundary witness.",
            Status.REFUTED,
            rejected.report.to_dict(),
        ),
        _scenario(
            "sampled-pass-abstention",
            "A below-tolerance error passes samples but is not promoted to proof.",
            Status.INCONCLUSIVE,
            abstained.report.to_dict(),
        ),
    ]
    abstention_events = abstained.report.evidence_events
    checks = {
        "exact_candidate_proved": repaired.report.status is Status.PROVED,
        "boundary_defect_refuted": rejected.report.status is Status.REFUTED
        and rejected.report.witness is not None,
        "sampled_pass_abstained": abstained.report.status is Status.INCONCLUSIVE
        and any(event.kind is EvidenceKind.EMPIRICAL_PASS for event in abstention_events),
        "repair_parent_preserved": (
            trace.evaluations[1].proposal.parent_proposal_id
            == trace.evaluations[0].proposal.proposal_id
        ),
    }

    return {
        "quickstart_version": QUICKSTART_VERSION,
        "mode": "offline-deterministic-fixture",
        "problem": {
            "id": "heat-classical-01",
            "name": trusted_case.problem.name,
            "solution_semantics": "classical_strong",
        },
        "configuration": {
            "tolerance": _TOLERANCE,
            "samples_per_axis": _SAMPLES_PER_AXIS,
            "symbolic_timeout_seconds": None,
            "max_expression_ops": _MAX_EXPRESSION_OPS,
        },
        "evidence_rule": "Empirical sampling may refute but never proves a candidate.",
        "scenarios": scenarios,
        "agent_trace": trace.to_dict(),
        "checks": checks,
        "passed": all(checks.values()) and all(item["passed"] for item in scenarios),
    }


def render_quickstart(payload: Mapping[str, object]) -> str:
    """Render a compact human-readable view of :func:`run_quickstart`."""

    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list):
        raise TypeError("quickstart scenarios must be a list")

    lines = ["PDECert offline quickstart", ""]
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise TypeError("quickstart scenario entries must be objects")
        report = scenario["report"]
        if not isinstance(report, dict):
            raise TypeError("quickstart reports must be objects")
        evidence = report["decision_evidence"] or "no decisive evidence"
        lines.append(f"[{index}/4] {scenario['id']}: {report['status']} ({evidence})")
        witness = report["witness"]
        if isinstance(witness, dict):
            lines.append(
                f"      witness: {witness['constraint']} at {witness['point']} "
                f"with residual {witness['residual']}"
            )

    trace = payload["agent_trace"]
    if not isinstance(trace, dict) or not isinstance(trace.get("evaluations"), list):
        raise TypeError("quickstart agent trace must contain evaluations")
    trace_statuses = [item["verification"]["status"] for item in trace["evaluations"]]
    lines.extend(
        [
            f"[4/4] agent-repair-trace: {' -> '.join(trace_statuses)}",
            "",
            str(payload["evidence_rule"]),
            (
                "PASS: all expected outcomes were reproduced."
                if payload["passed"]
                else "FAIL: at least one expected outcome changed."
            ),
            "Use `pdecert quickstart --json` for the complete reports and provenance.",
        ]
    )
    return "\n".join(lines) + "\n"
