"""Record a small proposal, counterexample, and repair loop."""

import json

import sympy as sp

from pdecert import (
    AgentProposal,
    AgentTrace,
    SymbolicCandidate,
    case_from_dict,
    evaluate_agent_proposal,
)


def main() -> None:
    x, t = sp.symbols("x t", real=True)
    exact = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    trusted_case = case_from_dict(
        {
            "schema_version": 3,
            "name": "heat equation",
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
            "fields": {"u": str(exact)},
        }
    )

    first = AgentProposal(
        "attempt-1",
        "example-agent/model-v1",
        SymbolicCandidate.from_expressions({"u": exact + t}),
        raw_output="u(x,t) = exp(-pi^2 t) sin(pi x) + t",
    )
    first_evaluation = evaluate_agent_proposal(trusted_case, first)

    repaired = AgentProposal(
        "attempt-2",
        "example-agent/model-v1",
        SymbolicCandidate.from_expressions({"u": exact}),
        raw_output="u(x,t) = exp(-pi^2 t) sin(pi x)",
        parent_proposal_id="attempt-1",
    )
    repaired_evaluation = evaluate_agent_proposal(trusted_case, repaired)

    trace = AgentTrace(
        "heat-repair-run-01",
        "heat-classical-01",
        (first_evaluation, repaired_evaluation),
    )
    print(json.dumps(trace.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
