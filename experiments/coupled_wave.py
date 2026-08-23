"""Exercise PDECert on a two-field first-order wave system."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from pdecert import Constraint, Problem, VerificationCase, verify


x, t = sp.symbols("x t", real=True)


def build_case(*, perturb_second_field: bool = False) -> VerificationCase:
    """Build an exact or deliberately perturbed two-field candidate."""

    first = sp.sin(sp.pi * x) * sp.cos(sp.pi * t)
    second = sp.cos(sp.pi * x) * sp.sin(sp.pi * t)
    if perturb_second_field:
        second += x * t / 10

    problem = Problem(
        name="coupled first-order wave system",
        variables=(x, t),
        domains={x: (0.0, 1.0), t: (0.0, 1.0)},
        pde_residuals=(
            Constraint(
                "u_t - v_x",
                sp.diff(first, t) - sp.diff(second, x),
                "D(u, t) - D(v, x)",
            ),
            Constraint(
                "v_t - u_x",
                sp.diff(second, t) - sp.diff(first, x),
                "D(v, t) - D(u, x)",
            ),
        ),
        conditions=(
            Constraint(
                "u initial condition",
                first.subs(t, 0) - sp.sin(sp.pi * x),
                "At(u, t, 0) - sin(pi*x)",
            ),
            Constraint("v initial condition", second.subs(t, 0), "At(v, t, 0)"),
            Constraint("u left boundary", first.subs(x, 0), "At(u, x, 0)"),
            Constraint("u right boundary", first.subs(x, 1), "At(u, x, 1)"),
        ),
    )
    return VerificationCase(problem, (first, second), ("u", "v"))


def run() -> list[dict[str, object]]:
    rows = []
    for name, perturbed in (("exact", False), ("perturbed_v", True)):
        case = build_case(perturb_second_field=perturbed)
        rows.append(
            {
                "case": name,
                "expected_valid": not perturbed,
                "fields": list(case.field_names),
                "report": verify(case.problem, case.candidate_fields).to_dict(),
            }
        )
    return rows


def main() -> None:
    output = Path(__file__).parents[1] / "results" / "coupled_wave.json"
    output.write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n")
    print(f"Detailed results: {output}")


if __name__ == "__main__":
    main()
