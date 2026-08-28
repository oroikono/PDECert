"""Bind symbolic and callable candidates to one candidate-free heat template."""

from pathlib import Path

import torch

from pdecert import (
    CallableCandidate,
    bind_symbolic_candidate,
    compile_autodiff_problem,
    load_template,
    verify,
    verify_artifact,
)


def exact_heat(points):
    x, t = points[:, 0:1], points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def main() -> None:
    template = load_template(Path(__file__).with_name("heat-template.json"))

    symbolic_case = bind_symbolic_candidate(
        template,
        {"u": "exp(-pi**2*t)*sin(pi*x)"},
    )
    symbolic_report = verify(symbolic_case.problem, symbolic_case.candidate_fields)

    callable_problem = compile_autodiff_problem(template)
    callable_report = verify_artifact(
        callable_problem,
        CallableCandidate.from_mapping({"u": exact_heat}, dtype="float64"),
        tolerance=1e-9,
    )

    print(f"symbolic: {symbolic_report.status.value} ({symbolic_report.decision_evidence.value})")
    print(f"callable: {callable_report.status.value} (sampled pass is not proof)")


if __name__ == "__main__":
    main()
