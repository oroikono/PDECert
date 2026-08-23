"""Conservative checks for symbolic differential-equation candidates.

The verifier uses three outcomes: proved, refuted, or inconclusive. Numerical
sampling may refute a candidate by producing a witness, but it never proves one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from typing import Iterable

import mpmath
import sympy as sp


class Status(str, Enum):
    """Possible outcomes of a verification attempt."""

    PROVED = "PROVED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Constraint:
    """A named residual that should be identically zero."""

    name: str
    residual: sp.Expr


@dataclass(frozen=True)
class Problem:
    """A fully instantiated symbolic PDE verification problem."""

    name: str
    variables: tuple[sp.Symbol, ...]
    domains: dict[sp.Symbol, tuple[float, float]]
    pde_residuals: tuple[Constraint, ...]
    conditions: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        missing = set(self.variables) - set(self.domains)
        extra = set(self.domains) - set(self.variables)
        if missing or extra:
            raise ValueError("domains must contain exactly the declared variables")
        if not self.pde_residuals and not self.conditions:
            raise ValueError("a problem must contain at least one verification constraint")
        for variable, (lower, upper) in self.domains.items():
            if lower >= upper:
                raise ValueError(f"invalid domain for {variable}: lower bound must be smaller")


@dataclass(frozen=True)
class Witness:
    """A concrete reason why a candidate was refuted."""

    constraint: str
    point: dict[str, float | str]
    residual: float | str
    reason: str


@dataclass
class Report:
    """Machine-readable result returned by :func:`verify`."""

    status: Status
    exact_checks: dict[str, str] = field(default_factory=dict)
    witness: Witness | None = None
    max_sampled_residual: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def _is_zero(expr: sp.Expr) -> bool | None:
    """Decide whether an expression is zero when simplification is conclusive."""

    simplified = sp.trigsimp(sp.cancel(sp.simplify(expr)))
    if simplified == 0 or simplified.is_zero is True:
        return True
    if simplified.is_zero is False and not simplified.free_symbols:
        return False
    return None


def _interior_points(lower: float, upper: float, count: int) -> list[float]:
    fractions = (0.113, 0.271, 0.419, 0.613, 0.787, 0.937)
    return [lower + (upper - lower) * fractions[index % len(fractions)] for index in range(count)]


def _numeric_value(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    values: tuple[float, ...],
) -> float:
    function = sp.lambdify(variables, expr, modules="mpmath")
    value = function(*values)
    if isinstance(value, mpmath.mpc):
        if abs(value.imag) > 1e-20:
            return float("inf")
        value = value.real
    return float(abs(value))


def _find_singularity(
    problem: Problem,
    expressions: Iterable[sp.Expr],
) -> tuple[Witness | None, bool]:
    """Search for domain singularities and report whether the search was complete."""

    complete = True
    for expr in expressions:
        for variable in problem.variables:
            lower, upper = problem.domains[variable]
            try:
                points = sp.singularities(expr, variable)
            except (NotImplementedError, ValueError):
                complete = False
                continue
            if not isinstance(points, sp.Set):
                complete = False
                continue
            try:
                concrete_points = list(points)
            except (TypeError, NotImplementedError):
                complete = False
                continue
            for point in concrete_points:
                if point.is_real is False or point.free_symbols:
                    if point.free_symbols:
                        complete = False
                    continue
                numeric = float(sp.N(point, 30))
                if lower <= numeric <= upper:
                    return (
                        Witness(
                            constraint="candidate domain",
                            point={str(variable): numeric},
                            residual="undefined",
                            reason=f"candidate has a singularity at {variable}={sp.sstr(point)}",
                        ),
                        complete,
                    )
    return None, complete


def verify(
    problem: Problem,
    candidate_expressions: Iterable[sp.Expr],
    *,
    tolerance: float = 1e-9,
    samples_per_axis: int = 5,
) -> Report:
    """Verify residuals and conditions for a symbolic candidate.

    Exact symbolic identities can prove the current obligations. Off-grid
    numerical evaluation is used only to find counterexamples when symbolic
    checking is inconclusive.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one")

    expressions = tuple(candidate_expressions)
    constraints = problem.pde_residuals + problem.conditions
    report = Report(status=Status.INCONCLUSIVE)

    singularity, domain_check_complete = _find_singularity(problem, expressions)
    if singularity is not None:
        report.status = Status.REFUTED
        report.witness = singularity
        return report

    decisions: list[bool | None] = []
    for constraint in constraints:
        decision = _is_zero(constraint.residual)
        decisions.append(decision)
        report.exact_checks[constraint.name] = (
            "identity" if decision is True else "nonzero" if decision is False else "unknown"
        )

    if domain_check_complete and all(decision is True for decision in decisions):
        report.status = Status.PROVED
        return report

    axes = [
        _interior_points(*problem.domains[variable], samples_per_axis)
        for variable in problem.variables
    ]
    max_residual = 0.0
    for constraint in constraints:
        for values in product(*axes):
            try:
                residual = _numeric_value(constraint.residual, problem.variables, values)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                residual = float("inf")
            max_residual = max(max_residual, residual)
            if residual > tolerance:
                report.status = Status.REFUTED
                report.max_sampled_residual = max_residual
                report.witness = Witness(
                    constraint=constraint.name,
                    point={
                        str(variable): float(value)
                        for variable, value in zip(problem.variables, values)
                        if variable in constraint.residual.free_symbols
                    },
                    residual=residual,
                    reason="off-grid evaluation found a violated obligation",
                )
                return report

    report.max_sampled_residual = max_residual
    return report


def fixed_collocation_check(
    problem: Problem,
    *,
    include_conditions: bool,
    grid: dict[sp.Symbol, tuple[float, ...]],
    tolerance: float = 1e-9,
) -> tuple[bool, float]:
    """Baseline checker that accepts when a fixed finite grid passes."""

    constraints = problem.pde_residuals + (problem.conditions if include_conditions else ())
    axes = [grid[variable] for variable in problem.variables]
    max_residual = 0.0
    for constraint in constraints:
        for values in product(*axes):
            try:
                residual = _numeric_value(constraint.residual, problem.variables, values)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                residual = float("inf")
            max_residual = max(max_residual, residual)
    return max_residual <= tolerance, max_residual
