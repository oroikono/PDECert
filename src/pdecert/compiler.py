"""Compile trusted symbolic operator sources into callable residual problems.

The compiler is intentionally narrower than the schema parser. It reuses the
restricted expression sources retained by schema-v3 cases, but accepts only the
subset whose pointwise classical semantics can be represented faithfully by the
current PyTorch automatic-differentiation backend.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping

from .autodiff import AutodiffConstraint, AutodiffEvaluation, AutodiffProblem
from .schema import VerificationCase
from .templates import ProblemTemplate


_MAX_DERIVATIVE_ORDER = 8
_TORCH_FUNCTIONS = frozenset(
    {
        "Abs",
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "erf",
        "exp",
        "log",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
_NUMERIC_CONSTRUCTORS = frozenset({"Float", "Integer", "Rational"})
_OPERATORS = frozenset({"At", "D"})
_CONSTANTS = {"E": math.e, "pi": math.pi}


class OperatorCompileError(ValueError):
    """Raised when a valid symbolic case has no faithful callable lowering."""


def compile_autodiff_problem(case: VerificationCase | ProblemTemplate) -> AutodiffProblem:
    """Compile one case or candidate-free template into an autodiff problem.

    The trusted constraint sources become callable residual operators. A
    template contains no candidate; candidate expressions in a fully
    instantiated case are not executed or translated. A separately supplied
    :class:`~pdecert.CallableCandidate` is evaluated against the compiled
    problem.

    Current scope is parameter-free classical problems on rectangular domains.
    ``At(expr, coordinate, value)`` is lowered to an autodiff surface only when
    every occurrence in one constraint uses a consistent fixed value and the
    fixed coordinate does not escape that ``At`` scope. Unsupported constructs
    fail during compilation rather than changing verification semantics.
    """

    if isinstance(case, VerificationCase):
        name = case.problem.name
        parameters = {
            str(variable): assumptions
            for variable, assumptions in case.problem.parameter_assumptions.items()
        }
        variables = tuple(str(variable) for variable in case.problem.variables)
        domains = {
            str(variable): case.problem.domains[variable] for variable in case.problem.variables
        }
        field_names = frozenset(case.field_names)
        pde_sources = tuple(
            (constraint.name, constraint.source) for constraint in case.problem.pde_residuals
        )
        condition_sources = tuple(
            (constraint.name, constraint.source) for constraint in case.problem.conditions
        )
    elif isinstance(case, ProblemTemplate):
        name = case.name
        parameters = case.parameters
        variables = case.variables
        domains = case.domains
        field_names = frozenset(case.field_names)
        pde_sources = tuple(
            (constraint.name, constraint.expression) for constraint in case.pde_residuals
        )
        condition_sources = tuple(
            (constraint.name, constraint.expression) for constraint in case.conditions
        )
    else:
        raise TypeError("case must be a VerificationCase or ProblemTemplate")
    if parameters:
        names = ", ".join(sorted(parameters))
        raise OperatorCompileError(
            "callable lowering does not yet support parameter variables: " + names
        )

    variable_names = frozenset(variables)
    sources = tuple(source for _, source in pde_sources + condition_sources if source is not None)
    missing_fields = sorted(
        field
        for field in field_names
        if not any(_contains_name(source, field) for source in sources)
    )
    if missing_fields:
        raise OperatorCompileError(
            "trusted operator sources do not reference candidate field(s): "
            + ", ".join(missing_fields)
        )

    pde_residuals = tuple(
        _compile_constraint(
            constraint_name,
            source,
            variables=variable_names,
            fields=field_names,
            domains=domains,
        )
        for constraint_name, source in pde_sources
    )
    conditions = tuple(
        _compile_constraint(
            constraint_name,
            source,
            variables=variable_names,
            fields=field_names,
            domains=domains,
        )
        for constraint_name, source in condition_sources
    )
    return AutodiffProblem(
        name=name,
        variables=variables,
        domains=domains,
        pde_residuals=pde_residuals,
        conditions=conditions,
    )


def _compile_constraint(
    name: str,
    source: str | None,
    *,
    variables: frozenset[str],
    fields: frozenset[str],
    domains: Mapping[str, tuple[float, float]],
) -> AutodiffConstraint:
    if source is None:
        raise OperatorCompileError(f"constraint {name!r} has no retained operator source")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise OperatorCompileError(
            f"constraint {name!r} has invalid operator syntax: {error.msg}"
        ) from error

    _validate_node(tree, variables=variables, fields=fields, constraint=name)
    fixed_coordinates = _surface_bindings(
        tree,
        domains=domains,
        constraint=name,
    )
    for variable in fixed_coordinates:
        if _name_escapes_surface(tree, variable, fields=fields):
            raise OperatorCompileError(
                f"constraint {name!r} uses fixed coordinate {variable!r} outside its At expression"
            )

    def residual(evaluation: AutodiffEvaluation) -> object:
        value = _evaluate_node(
            tree.body,
            evaluation=evaluation,
            variables=variables,
            fields=fields,
        )
        return _tensor_like(evaluation, value)

    return AutodiffConstraint(
        name=name,
        residual=residual,
        fixed_coordinates=fixed_coordinates,
    )


def _validate_node(
    node: ast.AST,
    *,
    variables: frozenset[str],
    fields: frozenset[str],
    constraint: str,
) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(
            node.body,
            variables=variables,
            fields=fields,
            constraint=constraint,
        )
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _unsupported(constraint, "only numeric literals can be lowered")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _unsupported(constraint, "only finite numeric literals can be lowered")
        return
    if isinstance(node, ast.Name):
        if node.id not in variables | fields | _CONSTANTS.keys():
            raise _unsupported(constraint, f"unknown operator symbol: {node.id}")
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_node(
            node.operand,
            variables=variables,
            fields=fields,
            constraint=constraint,
        )
        return
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
    ):
        _validate_node(
            node.left,
            variables=variables,
            fields=fields,
            constraint=constraint,
        )
        _validate_node(
            node.right,
            variables=variables,
            fields=fields,
            constraint=constraint,
        )
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = node.func.id
        if node.keywords:
            raise _unsupported(constraint, "keyword arguments cannot be lowered")
        if function == "D":
            if len(node.args) not in {2, 3}:
                raise _unsupported(constraint, "D expects expression, variable, and optional order")
            if not isinstance(node.args[1], ast.Name) or node.args[1].id not in variables:
                raise _unsupported(constraint, "D variable must be a declared coordinate")
            order = 1 if len(node.args) == 2 else _integer_literal(node.args[2], constraint)
            if not 1 <= order <= _MAX_DERIVATIVE_ORDER:
                raise _unsupported(
                    constraint,
                    f"D order must be between 1 and {_MAX_DERIVATIVE_ORDER}",
                )
            _validate_node(
                node.args[0],
                variables=variables,
                fields=fields,
                constraint=constraint,
            )
            return
        if function == "At":
            if len(node.args) != 3:
                raise _unsupported(constraint, "At expects expression, variable, and value")
            if not isinstance(node.args[1], ast.Name) or node.args[1].id not in variables:
                raise _unsupported(constraint, "At variable must be a declared coordinate")
            _constant_number(node.args[2], constraint)
            _validate_node(
                node.args[0],
                variables=variables,
                fields=fields,
                constraint=constraint,
            )
            return
        if function in _TORCH_FUNCTIONS:
            if len(node.args) != 1:
                raise _unsupported(constraint, f"{function} expects one argument")
            _validate_node(
                node.args[0],
                variables=variables,
                fields=fields,
                constraint=constraint,
            )
            return
        if function in _NUMERIC_CONSTRUCTORS:
            _numeric_constructor(function, node.args, constraint)
            return
        if function in _OPERATORS:
            raise _unsupported(constraint, f"unsupported {function} form")
        raise _unsupported(constraint, f"function {function!r} has no callable lowering")
    raise _unsupported(
        constraint,
        f"operator syntax {type(node).__name__} has no callable lowering",
    )


def _surface_bindings(
    node: ast.AST,
    *,
    domains: Mapping[str, tuple[float, float]],
    constraint: str,
) -> dict[str, float]:
    bindings: dict[str, float] = {}
    for child in ast.walk(node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "At"
        ):
            continue
        variable = child.args[1].id
        value = _constant_number(child.args[2], constraint)
        previous = bindings.get(variable)
        if previous is not None and not math.isclose(previous, value, rel_tol=0.0, abs_tol=0.0):
            raise OperatorCompileError(
                f"constraint {constraint!r} uses incompatible At surfaces for "
                f"{variable!r}: {previous} and {value}"
            )
        lower, upper = domains[variable]
        if not lower <= value <= upper:
            raise OperatorCompileError(
                f"constraint {constraint!r} fixes {variable!r} outside its domain"
            )
        bindings[variable] = value
    return bindings


def _name_escapes_surface(
    node: ast.AST,
    fixed_variable: str,
    *,
    fields: frozenset[str],
) -> bool:
    def visit(current: ast.AST, covered: frozenset[str]) -> bool:
        if isinstance(current, ast.Name):
            return (
                current.id == fixed_variable or current.id in fields
            ) and fixed_variable not in covered
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Name):
            if current.func.id == "At":
                variable = current.args[1].id
                return visit(current.args[0], covered | {variable})
            if current.func.id == "D":
                variable = current.args[1].id
                return (variable == fixed_variable and fixed_variable not in covered) or visit(
                    current.args[0], covered
                )
        return any(visit(child, covered) for child in ast.iter_child_nodes(current))

    return visit(node, frozenset())


def _evaluate_node(
    node: ast.AST,
    *,
    evaluation: AutodiffEvaluation,
    variables: frozenset[str],
    fields: frozenset[str],
) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in fields:
            return evaluation.field(node.id)
        if node.id in variables:
            return evaluation.coordinate(node.id)
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(
            node.operand,
            evaluation=evaluation,
            variables=variables,
            fields=fields,
        )
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(
            node.left,
            evaluation=evaluation,
            variables=variables,
            fields=fields,
        )
        right = _evaluate_node(
            node.right,
            evaluation=evaluation,
            variables=variables,
            fields=fields,
        )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right
    if isinstance(node, ast.Call):
        function = node.func.id
        if function == "D":
            value = _evaluate_node(
                node.args[0],
                evaluation=evaluation,
                variables=variables,
                fields=fields,
            )
            order = 1 if len(node.args) == 2 else _integer_literal(node.args[2], "runtime")
            return evaluation.derivative_value(value, node.args[1].id, order=order)
        if function == "At":
            return _evaluate_node(
                node.args[0],
                evaluation=evaluation,
                variables=variables,
                fields=fields,
            )
        if function in _NUMERIC_CONSTRUCTORS:
            return _numeric_constructor(function, node.args, "runtime")
        value = _evaluate_node(
            node.args[0],
            evaluation=evaluation,
            variables=variables,
            fields=fields,
        )
        return _torch_function(function, _tensor_like(evaluation, value))
    raise RuntimeError(f"unvalidated operator node reached runtime: {type(node).__name__}")


def _torch_function(name: str, value: object) -> object:
    import torch

    function_name = "abs" if name == "Abs" else name
    if name == "erf":
        return torch.erf(value)
    return getattr(torch, function_name)(value)


def _tensor_like(evaluation: AutodiffEvaluation, value: object) -> object:
    import torch

    if torch.is_tensor(value):
        return value
    return evaluation.constant(value)


def _contains_name(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree))


def _integer_literal(node: ast.AST, constraint: str) -> int:
    value = _constant_number(node, constraint)
    if not float(value).is_integer():
        raise _unsupported(constraint, "expected an integer literal")
    return int(value)


def _constant_number(node: ast.AST, constraint: str) -> float:
    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
    ):
        value = float(node.value)
    elif isinstance(node, ast.Name) and node.id in _CONSTANTS:
        value = _CONSTANTS[node.id]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _constant_number(node.operand, constraint)
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
    ):
        left = _constant_number(node.left, constraint)
        right = _constant_number(node.right, constraint)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            value = left / right
        else:
            value = left**right
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        value = _numeric_constructor(node.func.id, node.args, constraint)
    else:
        raise _unsupported(constraint, "At values and derivative orders must be constant numbers")
    if not math.isfinite(value):
        raise _unsupported(constraint, "constant values must be finite")
    return value


def _numeric_constructor(
    name: str,
    arguments: list[ast.expr],
    constraint: str,
) -> float:
    if name in {"Float", "Integer"}:
        if len(arguments) != 1:
            raise _unsupported(constraint, f"{name} expects one numeric argument")
        value = _constant_number(arguments[0], constraint)
        return float(int(value)) if name == "Integer" else float(value)
    if name == "Rational":
        if len(arguments) != 2:
            raise _unsupported(constraint, "Rational expects numerator and denominator")
        numerator = _constant_number(arguments[0], constraint)
        denominator = _constant_number(arguments[1], constraint)
        if denominator == 0:
            raise _unsupported(constraint, "Rational denominator cannot be zero")
        return numerator / denominator
    raise _unsupported(constraint, f"function {name!r} is not a numeric constructor")


def _unsupported(constraint: str, message: str) -> OperatorCompileError:
    return OperatorCompileError(f"constraint {constraint!r}: {message}")
