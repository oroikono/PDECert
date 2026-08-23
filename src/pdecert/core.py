"""Conservative checks for symbolic differential-equation candidates.

The verifier uses three outcomes: proved, refuted, or inconclusive. Numerical
sampling may refute a candidate by producing a witness, but it never proves one.
"""

from __future__ import annotations

import math
import signal
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, TypeVar

import mpmath
import sympy as sp

if TYPE_CHECKING:
    from .checks import CheckerRegistry


_Result = TypeVar("_Result")
PARAMETER_ASSUMPTIONS = frozenset(
    {"integer", "negative", "nonnegative", "nonpositive", "nonzero", "positive"}
)


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
    source: str | None = None


@dataclass(frozen=True)
class Problem:
    """A fully instantiated symbolic PDE verification problem."""

    name: str
    variables: tuple[sp.Symbol, ...]
    domains: dict[sp.Symbol, tuple[float, float]]
    pde_residuals: tuple[Constraint, ...]
    conditions: tuple[Constraint, ...] = ()
    parameter_assumptions: dict[sp.Symbol, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(self.variables) - set(self.domains)
        extra = set(self.domains) - set(self.variables)
        if missing or extra:
            raise ValueError("domains must contain exactly the declared variables")
        if not self.pde_residuals and not self.conditions:
            raise ValueError("a problem must contain at least one verification constraint")
        unknown_parameters = set(self.parameter_assumptions) - set(self.variables)
        if unknown_parameters:
            names = ", ".join(sorted(str(item) for item in unknown_parameters))
            raise ValueError(f"parameter assumptions reference undeclared variables: {names}")
        for variable, (lower, upper) in self.domains.items():
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"invalid domain for {variable}: bounds must be finite and increasing"
                )

        for parameter, assumptions in self.parameter_assumptions.items():
            unknown = set(assumptions) - PARAMETER_ASSUMPTIONS
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported assumptions for {parameter}: {names}")
            signs = set(assumptions) & {"negative", "nonnegative", "nonpositive", "positive"}
            if len(signs) > 1:
                raise ValueError(f"conflicting sign assumptions for {parameter}")

            lower, upper = self.domains[parameter]
            if "positive" in assumptions and lower <= 0:
                raise ValueError(f"domain for positive parameter {parameter} must be above zero")
            if "nonnegative" in assumptions and lower < 0:
                raise ValueError(f"domain for nonnegative parameter {parameter} cannot be negative")
            if "negative" in assumptions and upper >= 0:
                raise ValueError(f"domain for negative parameter {parameter} must be below zero")
            if "nonpositive" in assumptions and upper > 0:
                raise ValueError(f"domain for nonpositive parameter {parameter} cannot be positive")
            if "nonzero" in assumptions and lower <= 0 <= upper:
                raise ValueError(f"domain for nonzero parameter {parameter} cannot include zero")
            if "integer" in assumptions and math.ceil(lower) > math.floor(upper):
                raise ValueError(f"domain for integer parameter {parameter} contains no integers")


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


def _parameter_points(problem: Problem, variable: sp.Symbol, count: int) -> list[float]:
    lower, upper = problem.domains[variable]
    assumptions = problem.parameter_assumptions[variable]
    if "integer" in assumptions:
        values = list(range(math.ceil(lower), math.floor(upper) + 1))
        if len(values) <= count:
            return [float(value) for value in values]
        if count == 1:
            return [float(values[len(values) // 2])]
        indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
        return [float(values[index]) for index in dict.fromkeys(indices)]

    if count == 1:
        return [(lower + upper) / 2]
    fractions = [0.0, 1.0, 0.113, 0.419, 0.787]
    if count > len(fractions):
        fractions.extend(
            fraction for index in range(1, count) if (fraction := index / count) not in fractions
        )
    return [lower + (upper - lower) * fraction for fraction in fractions[:count]]


def _sample_points(problem: Problem, variable: sp.Symbol, count: int) -> list[float]:
    if variable in problem.parameter_assumptions:
        return _parameter_points(problem, variable, count)
    return _interior_points(*problem.domains[variable], count)


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
    expressions: Iterable[tuple[str, sp.Expr]],
    timeout_seconds: float | None,
) -> tuple[Witness | None, bool, dict[str, str]]:
    """Search for domain singularities and report whether the search was complete."""

    complete = True
    incomplete_reasons: dict[str, str] = {}
    for field_name, expr in expressions:
        for variable in problem.variables:
            lower, upper = problem.domains[variable]
            check_name = f"{field_name} domain in {variable}"
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
                        constraint=f"{field_name} domain",
                        point={str(variable): numeric},
                        residual="undefined",
                        reason=f"{field_name} has a singularity at {variable}={sp.sstr(point)}",
                    ),
                    complete,
                    incomplete_reasons,
                )
    return None, complete, incomplete_reasons


def verify(
    problem: Problem,
    candidate_expressions: Iterable[sp.Expr] | Mapping[str, sp.Expr],
    *,
    tolerance: float = 1e-9,
    samples_per_axis: int = 5,
    symbolic_timeout: float | None = None,
    checker_registry: CheckerRegistry | None = None,
) -> Report:
    """Verify residuals and conditions for a symbolic candidate.

    Exact symbolic identities can prove the current obligations. Off-grid
    numerical evaluation is used only to find counterexamples when symbolic
    checking is inconclusive. Candidate expressions may be passed as a mapping
    to attach field names to domain diagnostics. ``symbolic_timeout`` applies
    separately to each domain and identity check; ``None`` leaves those
    operations unbounded.
    """

    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one")
    if symbolic_timeout is not None and (
        not math.isfinite(symbolic_timeout) or symbolic_timeout <= 0
    ):
        raise ValueError("symbolic_timeout must be finite and positive")

    if isinstance(candidate_expressions, Mapping):
        expressions = tuple(candidate_expressions.items())
        if any(not isinstance(name, str) or not name for name, _ in expressions):
            raise ValueError("candidate field names must be non-empty strings")
    else:
        expressions = tuple(
            (f"candidate[{index}]", expression)
            for index, expression in enumerate(candidate_expressions)
        )
    if not expressions:
        raise ValueError("at least one candidate expression is required")
    if any(not isinstance(expression, sp.Expr) for _, expression in expressions):
        raise TypeError("candidate expressions must be SymPy expressions")
    from .checks import CheckContext, default_checker_registry, run_checks

    context = CheckContext(
        problem=problem,
        candidate_fields=expressions,
        tolerance=tolerance,
        samples_per_axis=samples_per_axis,
        symbolic_timeout=symbolic_timeout,
    )
    return run_checks(context, checker_registry or default_checker_registry())


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
