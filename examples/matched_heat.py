"""Evaluate the same classical heat problem as symbolic and callable artifacts."""

import json

import sympy as sp
import torch

from pdecert import (
    AutodiffConstraint,
    AutodiffProblem,
    CallableCandidate,
    Constraint,
    EvaluationLane,
    LaneVerificationOptions,
    MatchedCase,
    Problem,
    SymbolicCandidate,
    verify_matched_case,
)


def callable_heat(points):
    x, t = points[:, 0:1], points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def build_case() -> MatchedCase:
    x, t = sp.symbols("x t", real=True)
    symbolic_field = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    symbolic_problem = Problem(
        "symbolic heat equation",
        (x, t),
        {x: (0.0, 1.0), t: (0.0, 1.0)},
        (Constraint("heat PDE", sp.diff(symbolic_field, t) - sp.diff(symbolic_field, x, 2)),),
        (
            Constraint("initial condition", symbolic_field.subs(t, 0) - sp.sin(sp.pi * x)),
            Constraint("left boundary", symbolic_field.subs(x, 0)),
            Constraint("right boundary", symbolic_field.subs(x, 1)),
        ),
    )
    callable_problem = AutodiffProblem(
        "callable heat equation",
        ("x", "t"),
        {"x": (0.0, 1.0), "t": (0.0, 1.0)},
        (
            AutodiffConstraint(
                "heat PDE",
                lambda value: value.derivative("u", "t") - value.derivative("u", "x", order=2),
            ),
        ),
        (
            AutodiffConstraint(
                "initial condition",
                lambda value: value.field("u") - torch.sin(torch.pi * value.coordinate("x")),
                {"t": 0.0},
            ),
            AutodiffConstraint("left boundary", lambda value: value.field("u"), {"x": 0.0}),
            AutodiffConstraint("right boundary", lambda value: value.field("u"), {"x": 1.0}),
        ),
    )
    return MatchedCase(
        "heat-classical-01",
        ("x", "t"),
        ("u",),
        "classical",
        (
            EvaluationLane(
                "symbolic",
                symbolic_problem,
                SymbolicCandidate.from_expressions({"u": symbolic_field}),
            ),
            EvaluationLane(
                "callable",
                callable_problem,
                CallableCandidate.from_mapping({"u": callable_heat}, dtype="float64"),
            ),
        ),
    )


def main() -> None:
    options = {
        "symbolic": LaneVerificationOptions(symbolic_timeout=2.0),
        "callable": LaneVerificationOptions(tolerance=1e-9),
    }
    report = verify_matched_case(build_case(), options=options)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
