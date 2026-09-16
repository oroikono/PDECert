"""Versioned JSON representation for fully instantiated verification cases."""

from __future__ import annotations

import ast
import json
import keyword
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.core.facts import InconsistentAssumptions
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from .core import PARAMETER_ASSUMPTIONS, Constraint, Problem


SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, SCHEMA_VERSION})
_MAX_DERIVATIVE_ORDER = 8


def _differentiate(
    expression: sp.Expr, variable: sp.Expr, order: sp.Expr = sp.Integer(1)
) -> sp.Expr:
    if not isinstance(variable, sp.Symbol):
        raise ValueError("D variable must be a declared symbol")
    if order.is_Integer is not True or order.is_positive is not True:
        raise ValueError("D order must be a positive integer")
    if int(order) > _MAX_DERIVATIVE_ORDER:
        raise ValueError(f"D order cannot exceed {_MAX_DERIVATIVE_ORDER}")
    return sp.diff(expression, variable, int(order))


def _evaluate_at(expression: sp.Expr, variable: sp.Expr, value: sp.Expr) -> sp.Expr:
    if not isinstance(variable, sp.Symbol):
        raise ValueError("At variable must be a declared symbol")
    return expression.subs(variable, value)


_FUNCTIONS: dict[str, Any] = {
    "Abs": sp.Abs,
    "At": _evaluate_at,
    "D": _differentiate,
    "Ei": sp.Ei,
    "Float": sp.Float,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "acos": sp.acos,
    "asin": sp.asin,
    "atan": sp.atan,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "erf": sp.erf,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}
_CONSTANTS: dict[str, sp.Expr] = {"E": sp.E, "pi": sp.pi}
_RESERVED_NAMES = set(_FUNCTIONS) | set(_CONSTANTS)


class SchemaError(ValueError):
    """Raised when a verification case does not follow the JSON schema."""


@dataclass(frozen=True)
class VerificationCase:
    """A problem together with the candidate expressions used for domain checks."""

    problem: Problem
    candidate_expressions: tuple[sp.Expr, ...]
    field_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expressions = tuple(self.candidate_expressions)
        if not expressions:
            raise ValueError("candidate_expressions must not be empty")
        names = tuple(self.field_names) or tuple(
            f"candidate_{index}" for index in range(len(expressions))
        )
        if len(names) != len(expressions):
            raise ValueError("field_names must match candidate_expressions")
        if len(set(names)) != len(names):
            raise ValueError("field_names must be unique")
        for name in names:
            if (
                not isinstance(name, str)
                or not name.isascii()
                or not name.isidentifier()
                or keyword.iskeyword(name)
            ):
                raise ValueError("field names must be ASCII identifiers")
        object.__setattr__(self, "candidate_expressions", expressions)
        object.__setattr__(self, "field_names", names)

    @property
    def candidate_fields(self) -> dict[str, sp.Expr]:
        """Return candidate expressions keyed by their declared field names."""

        return dict(zip(self.field_names, self.candidate_expressions))


def _error(path: str, message: str) -> SchemaError:
    return SchemaError(f"{path}: {message}")


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise _error(path, f"missing field(s): {', '.join(missing)}")
    if unknown:
        raise _error(path, f"unknown field(s): {', '.join(unknown)}")


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "expected a non-empty string")
    return value


def _validate_expression_node(node: ast.AST, names: set[str], path: str) -> None:
    if isinstance(node, ast.Expression):
        _validate_expression_node(node.body, names, path)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _error(path, "only finite numeric literals are allowed")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _error(path, "only finite numeric literals are allowed")
        return
    if isinstance(node, ast.Name):
        if node.id not in names and node.id not in _CONSTANTS:
            raise _error(path, f"unknown symbol: {node.id}")
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_expression_node(node.operand, names, path)
        return
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
    ):
        _validate_expression_node(node.left, names, path)
        _validate_expression_node(node.right, names, path)
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCTIONS:
            raise _error(path, f"unsupported function: {node.func.id}")
        if node.keywords:
            raise _error(path, "keyword arguments are not allowed")
        for argument in node.args:
            _validate_expression_node(argument, names, path)
        return
    raise _error(path, f"unsupported expression syntax: {type(node).__name__}")


def _parse_expression(
    value: object,
    *,
    symbols: Mapping[str, sp.Expr],
    path: str,
) -> sp.Expr:
    source = _text(value, path)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise _error(path, f"invalid expression syntax: {error.msg}") from error
    _validate_expression_node(tree, set(symbols), path)

    global_dict = {"__builtins__": {}, **_FUNCTIONS, **_CONSTANTS}
    try:
        expression = parse_expr(
            source,
            local_dict=dict(symbols),
            global_dict=global_dict,
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as error:
        raise _error(path, f"could not parse expression: {error}") from error
    if not isinstance(expression, sp.Expr):
        raise _error(path, "expression did not produce a symbolic scalar")
    return expression


def _parse_constraints(
    value: object,
    *,
    symbols: Mapping[str, sp.Expr],
    path: str,
) -> tuple[Constraint, ...]:
    if not isinstance(value, list):
        raise _error(path, "expected a list")
    constraints: list[Constraint] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        payload = _object(item, item_path)
        _exact_keys(payload, required={"name", "expression"}, path=item_path)
        source = _text(payload["expression"], f"{item_path}.expression")
        constraints.append(
            Constraint(
                _text(payload["name"], f"{item_path}.name"),
                _parse_expression(
                    source,
                    symbols=symbols,
                    path=f"{item_path}.expression",
                ),
                source,
            )
        )
    return tuple(constraints)


def case_from_dict(value: object) -> VerificationCase:
    """Validate and parse one supported verification case."""

    payload = _object(value, "$")
    if "schema_version" not in payload:
        raise _error("$", "missing field(s): schema_version")
    version = payload["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise _error("$.schema_version", f"expected one of: {supported}")

    fields = {
        "schema_version",
        "name",
        "variables",
        "domains",
        "pde_residuals",
        "conditions",
    }
    if version >= 2:
        fields.add("parameters")
    if version == 3:
        fields.add("fields")
    else:
        fields.add("candidate_expressions")
    _exact_keys(payload, required=fields, path="$")

    raw_variables = payload["variables"]
    if not isinstance(raw_variables, list) or not raw_variables:
        raise _error("$.variables", "expected a non-empty list")
    variable_names: list[str] = []
    for index, raw_name in enumerate(raw_variables):
        name = _text(raw_name, f"$.variables[{index}]")
        if not name.isascii() or not name.isidentifier() or keyword.iskeyword(name):
            raise _error(f"$.variables[{index}]", "expected an ASCII Python identifier")
        if name in _RESERVED_NAMES:
            raise _error(f"$.variables[{index}]", f"reserved name: {name}")
        if name in variable_names:
            raise _error(f"$.variables[{index}]", f"duplicate variable: {name}")
        variable_names.append(name)
    parameter_assumptions: dict[str, frozenset[str]] = {}
    if version >= 2:
        raw_parameters = _object(payload["parameters"], "$.parameters")
        unknown_parameters = set(raw_parameters) - set(variable_names)
        if unknown_parameters:
            names = ", ".join(sorted(str(item) for item in unknown_parameters))
            raise _error("$.parameters", f"unknown parameter(s): {names}")
        for name, raw_assumptions in raw_parameters.items():
            if not isinstance(name, str):
                raise _error("$.parameters", "parameter names must be strings")
            path = f"$.parameters.{name}"
            if not isinstance(raw_assumptions, list):
                raise _error(path, "expected a list of assumptions")
            assumptions: list[str] = []
            for index, raw_assumption in enumerate(raw_assumptions):
                assumption = _text(raw_assumption, f"{path}[{index}]")
                if assumption not in PARAMETER_ASSUMPTIONS:
                    raise _error(f"{path}[{index}]", f"unsupported assumption: {assumption}")
                if assumption in assumptions:
                    raise _error(f"{path}[{index}]", f"duplicate assumption: {assumption}")
                assumptions.append(assumption)
            parameter_assumptions[name] = frozenset(assumptions)

    symbols: dict[str, sp.Symbol] = {}
    for name in variable_names:
        assumptions = parameter_assumptions.get(name, frozenset())
        symbol_properties = {item: True for item in assumptions}
        symbol_properties["real"] = True
        try:
            symbols[name] = sp.Symbol(name, **symbol_properties)
        except InconsistentAssumptions as error:
            raise _error(f"$.parameters.{name}", "assumptions are inconsistent") from error

    raw_domains = _object(payload["domains"], "$.domains")
    domain_names = set(raw_domains)
    if domain_names != set(variable_names):
        missing = sorted(set(variable_names) - domain_names)
        unknown = sorted(domain_names - set(variable_names))
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown: {', '.join(unknown)}")
        raise _error("$.domains", "; ".join(detail))
    domains: dict[sp.Symbol, tuple[float, float]] = {}
    for name in variable_names:
        bounds = raw_domains[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise _error(f"$.domains.{name}", "expected [lower, upper]")
        if any(isinstance(bound, bool) or not isinstance(bound, (int, float)) for bound in bounds):
            raise _error(f"$.domains.{name}", "bounds must be numbers")
        lower, upper = (float(bound) for bound in bounds)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise _error(f"$.domains.{name}", "bounds must be finite and increasing")
        domains[symbols[name]] = (lower, upper)

    if version == 3:
        raw_fields = _object(payload["fields"], "$.fields")
        if not raw_fields:
            raise _error("$.fields", "expected at least one candidate field")
        field_names: list[str] = []
        candidates_list: list[sp.Expr] = []
        for raw_name, raw_expression in raw_fields.items():
            if (
                not isinstance(raw_name, str)
                or not raw_name.isascii()
                or not raw_name.isidentifier()
                or keyword.iskeyword(raw_name)
            ):
                raise _error("$.fields", "field names must be ASCII identifiers")
            if raw_name in symbols or raw_name in _RESERVED_NAMES:
                raise _error("$.fields", f"field name conflicts with a declared name: {raw_name}")
            field_names.append(raw_name)
            candidates_list.append(
                _parse_expression(
                    raw_expression,
                    symbols=symbols,
                    path=f"$.fields.{raw_name}",
                )
            )
        candidates = tuple(candidates_list)
    else:
        raw_candidates = payload["candidate_expressions"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _error("$.candidate_expressions", "expected a non-empty list")
        candidates = tuple(
            _parse_expression(item, symbols=symbols, path=f"$.candidate_expressions[{index}]")
            for index, item in enumerate(raw_candidates)
        )
        field_names = [f"candidate_{index}" for index in range(len(candidates))]

    constraint_symbols: dict[str, sp.Expr] = dict(symbols)
    if version == 3:
        constraint_symbols.update(dict(zip(field_names, candidates)))
    pde_residuals = _parse_constraints(
        payload["pde_residuals"],
        symbols=constraint_symbols,
        path="$.pde_residuals",
    )
    conditions = _parse_constraints(
        payload["conditions"],
        symbols=constraint_symbols,
        path="$.conditions",
    )
    if not pde_residuals and not conditions:
        raise _error("$", "at least one PDE residual or condition is required")
    seen_constraints: set[str] = set()
    for constraint in pde_residuals + conditions:
        if constraint.name in seen_constraints:
            raise _error("$", f"duplicate constraint name: {constraint.name}")
        seen_constraints.add(constraint.name)

    try:
        problem = Problem(
            name=_text(payload["name"], "$.name"),
            variables=tuple(symbols[name] for name in variable_names),
            domains=domains,
            pde_residuals=pde_residuals,
            conditions=conditions,
            parameter_assumptions={
                symbols[name]: assumptions for name, assumptions in parameter_assumptions.items()
            },
        )
    except ValueError as error:
        raise _error("$", str(error)) from error
    return VerificationCase(problem, candidates, tuple(field_names))


def case_to_dict(case: VerificationCase) -> dict[str, object]:
    """Convert a verification case into the latest JSON representation."""

    variable_names = [str(variable) for variable in case.problem.variables]

    def constraints(items: tuple[Constraint, ...]) -> list[dict[str, str]]:
        return [
            {"name": item.name, "expression": item.source or sp.sstr(item.residual)}
            for item in items
        ]

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "name": case.problem.name,
        "variables": variable_names,
        "domains": {
            str(variable): list(case.problem.domains[variable])
            for variable in case.problem.variables
        },
        "parameters": {
            str(variable): sorted(assumptions)
            for variable, assumptions in case.problem.parameter_assumptions.items()
        },
        "pde_residuals": constraints(case.problem.pde_residuals),
        "conditions": constraints(case.problem.conditions),
        "fields": {
            name: sp.sstr(expression)
            for name, expression in zip(case.field_names, case.candidate_expressions)
        },
    }
    validated = case_from_dict(payload)
    symbol_map = dict(zip(validated.problem.variables, case.problem.variables))
    original_constraints = case.problem.pde_residuals + case.problem.conditions
    validated_constraints = validated.problem.pde_residuals + validated.problem.conditions
    for original, rendered in zip(original_constraints, validated_constraints):
        if original.source is None:
            continue
        rendered_residual = rendered.residual.xreplace(symbol_map)
        if sp.simplify(rendered_residual - original.residual) != 0:
            raise _error("$", f"constraint source does not match residual: {original.name}")
    return payload


def load_case(path: str | Path) -> VerificationCase:
    """Load and validate a verification case from a JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise SchemaError(f"{source}: invalid JSON: {error.msg}") from error
    return case_from_dict(payload)


def dump_case(case: VerificationCase, path: str | Path) -> None:
    """Write a verification case as deterministic, readable JSON."""

    Path(path).write_text(json.dumps(case_to_dict(case), indent=2, sort_keys=True) + "\n")
