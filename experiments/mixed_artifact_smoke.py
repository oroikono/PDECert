"""Smoke comparison across symbolic and PyTorch solution artifacts.

This four-case suite exercises the shared interface. It is not a population-level
benchmark and should not be reported as evidence of broad model performance.
"""

from __future__ import annotations

import json

import sympy as sp
import torch

from examples.autodiff_heat import build_problem, exact_heat, perturbed_heat
from pdecert import (
    CallableCandidate,
    Constraint,
    Problem,
    SymbolicCandidate,
    verify_artifact,
)


def symbolic_case(*, perturbed: bool):
    x, t = sp.symbols("x t", real=True)
    candidate = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    if perturbed:
        candidate += sp.Rational(1, 10) * t * x * (1 - x)
    problem = Problem(
        name="symbolic heat equation",
        variables=(x, t),
        domains={x: (0.0, 1.0), t: (0.0, 1.0)},
        pde_residuals=(Constraint("heat PDE", sp.diff(candidate, t) - sp.diff(candidate, x, 2)),),
        conditions=(
            Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
            Constraint("left boundary", candidate.subs(x, 0)),
            Constraint("right boundary", candidate.subs(x, 1)),
        ),
    )
    return problem, SymbolicCandidate.from_expressions({"u": candidate})


def run_suite() -> dict[str, object]:
    """Run two symbolic and two differentiable callable cases."""

    records = []
    for name, perturbed in (("symbolic-exact", False), ("symbolic-perturbed", True)):
        problem, artifact = symbolic_case(perturbed=perturbed)
        report = verify_artifact(problem, artifact)
        records.append({"name": name, "expected_valid": not perturbed, **report.to_dict()})

    problem = build_problem()
    for name, field, expected_valid in (
        ("callable-exact", exact_heat, True),
        ("callable-perturbed", perturbed_heat, False),
    ):
        artifact = CallableCandidate.from_mapping({"u": field}, dtype="float64")
        report = verify_artifact(problem, artifact, tolerance=1e-9)
        records.append({"name": name, "expected_valid": expected_valid, **report.to_dict()})

    return {
        "suite": "mixed-artifact-smoke-v1",
        "disclaimer": "interface smoke suite; not a model-performance benchmark",
        "torch_version": torch.__version__,
        "records": records,
    }


def main() -> None:
    print(json.dumps(run_suite(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
