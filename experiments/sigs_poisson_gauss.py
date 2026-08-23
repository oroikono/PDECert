"""Check the boundary obligations of the Poisson--Gauss candidate reported in SIGS."""

from __future__ import annotations

import json

import sympy as sp

from pdecert import Constraint, Problem, verify


def build_probe() -> tuple[Problem, sp.Expr]:
    x, y = sp.symbols("x y", real=True)
    r1_squared = (x - sp.Rational(3, 10)) ** 2 + (y - sp.Rational(1, 2)) ** 2
    r2_squared = (x - sp.Rational(7, 10)) ** 2 + (y - sp.Rational(1, 5)) ** 2
    candidate = -sp.Rational(1, 100) * (
        sp.log(sp.sqrt(r1_squared))
        - sp.Ei(-r1_squared / sp.Rational(1, 50)) / 2
        + sp.log(sp.sqrt(r2_squared))
        - sp.Ei(-r2_squared / sp.Rational(1, 50)) / 2
    )
    problem = Problem(
        name="sigs_poisson_gauss_candidate",
        variables=(x, y),
        domains={x: (0.0, 1.0), y: (0.0, 1.0)},
        pde_residuals=(),
        conditions=(
            Constraint("x=0 homogeneous boundary", candidate.subs(x, 0)),
            Constraint("x=1 homogeneous boundary", candidate.subs(x, 1)),
            Constraint("y=0 homogeneous boundary", candidate.subs(y, 0)),
            Constraint("y=1 homogeneous boundary", candidate.subs(y, 1)),
        ),
    )
    return problem, candidate


def main() -> None:
    problem, candidate = build_probe()
    print(json.dumps(verify(problem, (candidate,)).to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
