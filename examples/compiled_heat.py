"""Evaluate symbolic and callable heat fields from one trusted operator source."""

from pathlib import Path

import torch

from pdecert import (
    CallableCandidate,
    SymbolicCandidate,
    compile_autodiff_problem,
    load_case,
    verify_artifact,
)


def exact_heat(points):
    x, t = points[:, 0:1], points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def main() -> None:
    case = load_case(Path(__file__).with_name("exact_heat.json"))

    symbolic = verify_artifact(
        case.problem,
        SymbolicCandidate.from_expressions(case.candidate_fields),
    )
    callable_problem = compile_autodiff_problem(case)
    callable_report = verify_artifact(
        callable_problem,
        CallableCandidate.from_mapping({"u": exact_heat}, dtype="float64"),
        tolerance=1e-9,
    )

    print(f"symbolic: {symbolic.status.value} ({symbolic.decision_evidence.value})")
    print(f"callable: {callable_report.status.value} (sampled pass is not proof)")


if __name__ == "__main__":
    main()
