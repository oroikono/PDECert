"""Label-gated benchmark evaluation for candidate-corpus records."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import sympy as sp

from .core import Problem, Status, _run_bounded, fixed_collocation_check, verify
from .corpus import validate_corpus
from .schema import VerificationCase, case_from_dict


BENCHMARK_VERSION = 1
OUTCOMES = frozenset({"accept", "inconclusive", "reject"})


class BenchmarkError(ValueError):
    """Raised when a corpus is not ready for benchmark evaluation."""


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def corpus_sha256(corpus: object) -> str:
    """Return the canonical digest used to bind reports to labeled corpora."""

    encoded = json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_version() -> str:
    try:
        return version("pdecert")
    except PackageNotFoundError:
        return "source checkout"


def _direct_sympy_zero(expression: sp.Expr) -> bool | None:
    simplified = sp.trigsimp(sp.cancel(sp.simplify(expression)))
    if simplified == 0 or simplified.is_zero is True:
        return True
    if simplified.is_zero is False and not simplified.free_symbols:
        return False
    return None


def _grid(problem: Problem, points_per_axis: int) -> dict[sp.Symbol, tuple[float, ...]]:
    grid: dict[sp.Symbol, tuple[float, ...]] = {}
    for variable in problem.variables:
        lower, upper = problem.domains[variable]
        assumptions = problem.parameter_assumptions.get(variable, frozenset())
        if "integer" in assumptions:
            values = list(range(math.ceil(lower), math.floor(upper) + 1))
            if len(values) > points_per_axis:
                indices = {
                    round(index * (len(values) - 1) / (points_per_axis - 1))
                    for index in range(points_per_axis)
                }
                values = [values[index] for index in sorted(indices)]
            grid[variable] = tuple(float(value) for value in values)
        else:
            step = (upper - lower) / (points_per_axis - 1)
            grid[variable] = tuple(lower + index * step for index in range(points_per_axis))
    return grid


def _symbolic_residual_outcome(
    problem: Problem,
    symbolic_timeout: float | None,
) -> tuple[str, dict[str, str]]:
    decisions: list[bool | None] = []
    details: dict[str, str] = {}
    for constraint in problem.pde_residuals + problem.conditions:
        decision, error = _run_bounded(
            lambda residual=constraint.residual: _direct_sympy_zero(residual),
            symbolic_timeout,
        )
        decisions.append(decision)
        details[constraint.name] = (
            f"inconclusive: {error}"
            if error is not None
            else "zero"
            if decision is True
            else "nonzero"
            if decision is False
            else "inconclusive"
        )
    if all(decision is True for decision in decisions):
        return "accept", details
    if any(decision is False for decision in decisions):
        return "reject", details
    return "inconclusive", details


def _clean_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _metrics(records: list[dict[str, object]]) -> dict[str, object]:
    scored = [record for record in records if record["truth"] in {"valid", "invalid"}]
    valid = [record for record in scored if record["truth"] == "valid"]
    invalid = [record for record in scored if record["truth"] == "invalid"]
    decisive = [record for record in scored if record["outcome"] != "inconclusive"]
    correct = [
        record
        for record in scored
        if (record["truth"], record["outcome"]) in {("valid", "accept"), ("invalid", "reject")}
    ]
    false_acceptances = [record for record in invalid if record["outcome"] == "accept"]
    false_rejections = [record for record in valid if record["outcome"] == "reject"]
    inconclusive = [record for record in scored if record["outcome"] == "inconclusive"]
    witnessed_invalid = [
        record
        for record in invalid
        if record["outcome"] == "reject" and record.get("witness") is not None
    ]
    return {
        "accuracy": _rate(len(correct), len(scored)),
        "correct_count": len(correct),
        "decisive_accuracy": _rate(len(correct), len(decisive)),
        "decisive_count": len(decisive),
        "false_acceptance_count": len(false_acceptances),
        "false_acceptance_rate": _rate(len(false_acceptances), len(invalid)),
        "false_rejection_count": len(false_rejections),
        "false_rejection_rate": _rate(len(false_rejections), len(valid)),
        "inconclusive_count": len(inconclusive),
        "inconclusive_rate": _rate(len(inconclusive), len(scored)),
        "invalid_count": len(invalid),
        "invalid_witness_count": len(witnessed_invalid),
        "invalid_witness_rate": _rate(len(witnessed_invalid), len(invalid)),
        "scored_count": len(scored),
        "valid_count": len(valid),
    }


def _run_method(
    records: list[Mapping[str, object]],
    check: Callable[[VerificationCase], dict[str, object]],
) -> dict[str, object]:
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    for record in records:
        case = case_from_dict(record["case"])
        item_started = time.perf_counter()
        result = check(case)
        elapsed = time.perf_counter() - item_started
        outcome = result.get("outcome")
        if outcome not in OUTCOMES:
            raise BenchmarkError(f"checker returned an unsupported outcome: {outcome}")
        results.append(
            {
                "id": record["id"],
                "outcome": outcome,
                "runtime_seconds": elapsed,
                "truth": record["annotation"]["verdict"],
                **{key: value for key, value in result.items() if key != "outcome"},
            }
        )
    return {
        "metrics": _metrics(results),
        "records": _clean_json(results),
        "runtime_seconds": time.perf_counter() - started,
    }


def evaluate_corpus(
    corpus: object,
    *,
    points_per_axis: int = 5,
    tolerance: float = 1e-9,
    symbolic_timeout: float | None = 2.0,
) -> dict[str, Any]:
    """Evaluate three checkers, refusing any corpus with pending annotations."""

    validate_corpus(corpus)
    if not isinstance(corpus, Mapping):
        raise BenchmarkError("corpus must be an object")
    if isinstance(points_per_axis, bool) or not isinstance(points_per_axis, int):
        raise BenchmarkError("points_per_axis must be an integer")
    if points_per_axis < 2:
        raise BenchmarkError("points_per_axis must be at least two")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise BenchmarkError("tolerance must be a number")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise BenchmarkError("tolerance must be finite and positive")
    if symbolic_timeout is not None and (
        isinstance(symbolic_timeout, bool)
        or not isinstance(symbolic_timeout, (int, float))
        or not math.isfinite(symbolic_timeout)
        or symbolic_timeout <= 0
    ):
        raise BenchmarkError("symbolic_timeout must be finite and positive")

    records = corpus["records"]
    pending = [record["id"] for record in records if record["annotation"]["status"] == "pending"]
    if pending:
        raise BenchmarkError(
            "benchmark requires completed human labels; pending: " + ", ".join(pending)
        )
    scored = [
        record for record in records if record["annotation"]["verdict"] in {"valid", "invalid"}
    ]
    if not scored:
        raise BenchmarkError("benchmark requires at least one valid or invalid label")

    def fixed(case: VerificationCase) -> dict[str, object]:
        accepted, max_residual = fixed_collocation_check(
            case.problem,
            include_conditions=True,
            grid=_grid(case.problem, points_per_axis),
            tolerance=tolerance,
        )
        return {
            "outcome": "accept" if accepted else "reject",
            "max_residual": max_residual,
        }

    def symbolic(case: VerificationCase) -> dict[str, object]:
        outcome, details = _symbolic_residual_outcome(case.problem, symbolic_timeout)
        return {"outcome": outcome, "checks": details}

    def pdecert(case: VerificationCase) -> dict[str, object]:
        report = verify(
            case.problem,
            case.candidate_fields,
            tolerance=tolerance,
            samples_per_axis=points_per_axis,
            symbolic_timeout=symbolic_timeout,
        )
        outcome = {
            Status.PROVED: "accept",
            Status.REFUTED: "reject",
            Status.INCONCLUSIVE: "inconclusive",
        }[report.status]
        report_payload = report.to_dict()
        return {
            "outcome": outcome,
            "report": report_payload,
            "witness": report_payload["witness"],
        }

    methods = {
        "fixed_collocation": _run_method(scored, fixed),
        "pdecert": _run_method(scored, pdecert),
        "sympy_residual": _run_method(scored, symbolic),
    }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "configuration": {
            "method_order": ["fixed_collocation", "pdecert", "sympy_residual"],
            "points_per_axis": points_per_axis,
            "symbolic_timeout_seconds": symbolic_timeout,
            "timing_note": (
                "Single-process wall-clock timings are descriptive; method order and symbolic "
                "caches can affect them."
            ),
            "tolerance": tolerance,
        },
        "corpus": {
            "excluded_unclear": len(records) - len(scored),
            "name": corpus["name"],
            "sha256": corpus_sha256(corpus),
            "scored_records": len(scored),
            "total_records": len(records),
        },
        "environment": {
            "pdecert": _package_version(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sympy": sp.__version__,
        },
        "method_definitions": {
            "fixed_collocation": (
                "All PDE residuals and represented conditions on a uniform grid including "
                "domain endpoints; finite passing is treated as acceptance."
            ),
            "pdecert": (
                "Domain diagnostics, exact residual checks, and off-grid counterexample search; "
                "sampling can reject but cannot prove."
            ),
            "sympy_residual": (
                "Independent direct SymPy simplification of every represented residual and "
                "condition, with an inconclusive outcome when zero or nonzero cannot be decided."
            ),
        },
        "methods": methods,
    }
