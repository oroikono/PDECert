"""Conservative checks for symbolic differential-equation candidates.

The verifier uses three outcomes: proved, refuted, or inconclusive. Numerical
sampling may refute a candidate by producing a witness, but it never proves one.
"""

from __future__ import annotations

import math
import signal
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from typing import TypeVar

import mpmath
import sympy as sp


_Result = TypeVar("_Result")


class _DeadlineExceeded(Exception):
    pass


class _IncompleteCheck(Exception):
    pass


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
    incomplete_reasons: dict[str, str] = field(default_factory=dict)
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


def _deadline_supported() -> bool:
    return (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "setitimer")
    )


def _run_bounded(
    operation: Callable[[], _Result],
    timeout_seconds: float | None,
) -> tuple[_Result | None, str | None]:
    """Run one symbolic operation with an optional real-time deadline."""

    if timeout_seconds is None:
        try:
            return operation(), None
        except _IncompleteCheck as error:
            return None, str(error)
        except Exception as error:
            return None, f"symbolic check raised {type(error).__name__}: {error}"

    if not _deadline_supported():
        return None, "symbolic deadlines are unavailable outside the main thread on this platform"

    active_timer, _ = signal.getitimer(signal.ITIMER_REAL)
    if active_timer > 0:
        return None, "symbolic deadline was not started because another real-time timer is active"

    def raise_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise _DeadlineExceeded

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return operation(), None
    except _DeadlineExceeded:
        return None, f"symbolic check exceeded {timeout_seconds:g} seconds"
    except _IncompleteCheck as error:
        return None, str(error)
    except Exception as error:
        return None, f"symbolic check raised {type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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


def _domain_singularity(
    expr: sp.Expr,
    variable: sp.Symbol,
    lower: float,
    upper: float,
) -> tuple[sp.Expr, float] | None:
    points = sp.singularities(expr, variable)
    if not isinstance(points, sp.Set):
        raise _IncompleteCheck("singularity analysis returned an unsupported result")
    try:
        concrete_points = list(points)
    except (TypeError, NotImplementedError) as error:
        raise _IncompleteCheck("singularity set could not be enumerated") from error

    unresolved = False
    for point in concrete_points:
        if point.is_real is False:
            continue
        if point.free_symbols:
            unresolved = True
            continue
        numeric = float(sp.N(point, 30))
        if lower <= numeric <= upper:
            return point, numeric
    if unresolved:
        raise _IncompleteCheck("singularity location depends on unresolved symbols")
    return None


def _find_singularity(
    problem: Problem,
    expressions: Iterable[sp.Expr],
    timeout_seconds: float | None,
) -> tuple[Witness | None, bool, dict[str, str]]:
    """Search for domain singularities and report whether the search was complete."""

    complete = True
    incomplete_reasons: dict[str, str] = {}
    for expression_index, expr in enumerate(expressions):
        for variable in problem.variables:
            lower, upper = problem.domains[variable]
            check_name = f"candidate[{expression_index}] domain in {variable}"
            location, error = _run_bounded(
                lambda: _domain_singularity(expr, variable, lower, upper),
                timeout_seconds,
            )
            if error is not None:
                complete = False
                incomplete_reasons[check_name] = error
                continue
            if location is not None:
                point, numeric = location
                return (
                    Witness(
                        constraint="candidate domain",
                        point={str(variable): numeric},
                        residual="undefined",
                        reason=f"candidate has a singularity at {variable}={sp.sstr(point)}",
                    ),
                    complete,
                    incomplete_reasons,
                )
    return None, complete, incomplete_reasons


def verify(
    problem: Problem,
    candidate_expressions: Iterable[sp.Expr],
    *,
    tolerance: float = 1e-9,
    samples_per_axis: int = 5,
    symbolic_timeout: float | None = None,
) -> Report:
    """Verify residuals and conditions for a symbolic candidate.

    Exact symbolic identities can prove the current obligations. Off-grid
    numerical evaluation is used only to find counterexamples when symbolic
    checking is inconclusive. ``symbolic_timeout`` applies separately to each
    domain and identity check; ``None`` leaves those operations unbounded.
    """

    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one")
    if symbolic_timeout is not None and (
        not math.isfinite(symbolic_timeout) or symbolic_timeout <= 0
    ):
        raise ValueError("symbolic_timeout must be finite and positive")

    expressions = tuple(candidate_expressions)
    constraints = problem.pde_residuals + problem.conditions
    report = Report(status=Status.INCONCLUSIVE)

    singularity, domain_check_complete, domain_reasons = _find_singularity(
        problem,
        expressions,
        symbolic_timeout,
    )
    report.incomplete_reasons.update(domain_reasons)
    if singularity is not None:
        report.status = Status.REFUTED
        report.witness = singularity
        return report

    decisions: list[bool | None] = []
    for constraint in constraints:
        decision, error = _run_bounded(
            lambda: _is_zero(constraint.residual),
            symbolic_timeout,
        )
        decisions.append(decision)
        report.exact_checks[constraint.name] = (
            "identity" if decision is True else "nonzero" if decision is False else "unknown"
        )
        if error is not None:
            report.incomplete_reasons[constraint.name] = error
        elif decision is None:
            report.incomplete_reasons[constraint.name] = "symbolic simplification did not decide"

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
