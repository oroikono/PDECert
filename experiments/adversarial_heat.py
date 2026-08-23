"""Compare conservative verification with fixed collocation on the heat equation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from sympy.solvers.pde import checkpdesol

from pdecert import Constraint, Problem, fixed_collocation_check, verify


x, t, k = sp.symbols("x t k", real=True)
u = sp.Function("u")


@dataclass(frozen=True)
class Case:
    name: str
    expected_valid: bool
    explanation: str
    candidate: sp.Expr
    problem: Problem
    baseline_grid: dict[sp.Symbol, tuple[float, ...]]


def sympy_pde_check(case: Case) -> bool | None:
    """Run SymPy's PDE-only candidate check as a second baseline."""

    diffusivity = k if k in case.problem.variables else 1
    equation = sp.Eq(sp.diff(u(x, t), t), diffusivity * sp.diff(u(x, t), x, 2))
    try:
        decision, _ = checkpdesol(
            equation,
            sp.Eq(u(x, t), case.candidate),
            func=u(x, t),
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    return bool(decision)


def heat_problem(name: str, candidate: sp.Expr) -> Problem:
    return Problem(
        name=name,
        variables=(x, t),
        domains={x: (0.0, 1.0), t: (0.0, 1.0)},
        pde_residuals=(Constraint("heat PDE", sp.diff(candidate, t) - sp.diff(candidate, x, 2)),),
        conditions=(
            Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
            Constraint("left boundary", candidate.subs(x, 0)),
            Constraint("right boundary", candidate.subs(x, 1)),
        ),
    )


def parametric_heat_problem(name: str, candidate: sp.Expr) -> Problem:
    return Problem(
        name=name,
        variables=(x, t, k),
        domains={x: (0.0, 1.0), t: (0.0, 1.0), k: (0.2, 2.0)},
        pde_residuals=(
            Constraint("parametric heat PDE", sp.diff(candidate, t) - k * sp.diff(candidate, x, 2)),
        ),
        conditions=(
            Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
            Constraint("left boundary", candidate.subs(x, 0)),
            Constraint("right boundary", candidate.subs(x, 1)),
        ),
    )


def build_cases() -> list[Case]:
    grid_x = tuple(index / 4 for index in range(5))
    grid_t = tuple(index / 4 for index in range(5))
    heat_grid = {x: grid_x, t: grid_t}

    exact = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    equivalent = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * (x + 2))

    # q and q'' vanish at all five x coordinates used by the fixed grid.
    q = sp.prod((4 * x - index) ** 3 for index in range(5))
    grid_alias = exact + sp.Rational(1, 100_000) * t * q
    singular_alias = exact + sp.Rational(1, 100_000) * t * q / (3 * x - 1)
    parameter_trap = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    tiny_error = exact + sp.Rational(1, 10**14) * t * x * (1 - x)

    return [
        Case(
            "exact_heat_solution",
            True,
            "Canonical analytical solution.",
            exact,
            heat_problem("exact_heat_solution", exact),
            heat_grid,
        ),
        Case(
            "equivalent_expression",
            True,
            "Equivalent through sin(pi*(x+2)) = sin(pi*x).",
            equivalent,
            heat_problem("equivalent_expression", equivalent),
            heat_grid,
        ),
        Case(
            "pde_only_boundary_trap",
            False,
            "The PDE residual is zero, but the initial and right boundary conditions fail.",
            exact + x / 10,
            heat_problem("pde_only_boundary_trap", exact + x / 10),
            heat_grid,
        ),
        Case(
            "fixed_grid_alias",
            False,
            "A polynomial perturbation is invisible on every fixed collocation point.",
            grid_alias,
            heat_problem("fixed_grid_alias", grid_alias),
            heat_grid,
        ),
        Case(
            "hidden_singularity",
            False,
            "A grid-alias perturbation hides an interior pole at x=1/3.",
            singular_alias,
            heat_problem("hidden_singularity", singular_alias),
            heat_grid,
        ),
        Case(
            "single_parameter_trap",
            False,
            "The candidate solves the equation only at k=1, not over the declared interval.",
            parameter_trap,
            parametric_heat_problem("single_parameter_trap", parameter_trap),
            {x: grid_x, t: grid_t, k: (1.0,)},
        ),
        Case(
            "below_numeric_tolerance",
            False,
            "A real error is below tolerance and must not be reported as proved.",
            tiny_error,
            heat_problem("below_numeric_tolerance", tiny_error),
            heat_grid,
        ),
    ]


def run() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in build_cases():
        pde_only, pde_only_max = fixed_collocation_check(
            case.problem,
            include_conditions=False,
            grid=case.baseline_grid,
        )
        full_grid, full_grid_max = fixed_collocation_check(
            case.problem,
            include_conditions=True,
            grid=case.baseline_grid,
        )
        report = verify(case.problem, (case.candidate,))
        rows.append(
            {
                "case": case.name,
                "expected_valid": case.expected_valid,
                "explanation": case.explanation,
                "pde_only_fixed_grid_accepts": pde_only,
                "pde_only_max_residual": pde_only_max,
                "full_fixed_grid_accepts": full_grid,
                "full_fixed_max_residual": full_grid_max,
                "sympy_pde_accepts": sympy_pde_check(case),
                "pdecert": report.to_dict(),
            }
        )
    return rows


def main() -> None:
    rows = run()
    output = Path(__file__).parents[1] / "results" / "adversarial_heat.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    print(f"{'case':30} {'truth':7} {'PDE grid':9} {'full grid':10} {'SymPy':7} {'PDECert':12}")
    print("-" * 84)
    for row in rows:
        sympy_result = row["sympy_pde_accepts"]
        print(
            f"{row['case']:30} "
            f"{('valid' if row['expected_valid'] else 'wrong'):7} "
            f"{('ACCEPT' if row['pde_only_fixed_grid_accepts'] else 'reject'):9} "
            f"{('ACCEPT' if row['full_fixed_grid_accepts'] else 'reject'):10} "
            f"{('ACCEPT' if sympy_result else 'reject' if sympy_result is False else 'unknown'):7} "
            f"{row['pdecert']['status']:12}"
        )
    print(f"\nDetailed results: {output}")


if __name__ == "__main__":
    main()
