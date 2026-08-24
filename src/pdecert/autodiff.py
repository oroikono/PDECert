"""Conservative residual checks for differentiable callable candidates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from .artifacts import CallableCandidate
from .checks import CheckResult, CheckerRegistry, run_checks
from .core import Report, Witness


@dataclass(frozen=True)
class AutodiffConstraint:
    """A residual operator evaluated on an autodiff context.

    ``fixed_coordinates`` restricts an obligation to an initial, boundary, or
    interface surface. Unfixed variables are sampled in the domain interior.
    """

    name: str
    residual: Callable[[AutodiffEvaluation], object]
    fixed_coordinates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("autodiff constraint names must be non-empty")
        if not callable(self.residual):
            raise TypeError("autodiff constraint residuals must be callable")
        normalized = dict(self.fixed_coordinates)
        if any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError("fixed coordinate names must be non-empty strings")
        if any(not math.isfinite(value) for value in normalized.values()):
            raise ValueError("fixed coordinate values must be finite")
        object.__setattr__(self, "fixed_coordinates", MappingProxyType(normalized))


@dataclass(frozen=True)
class AutodiffProblem:
    """A rectangular PDE problem whose obligations use autodiff operators."""

    name: str
    variables: tuple[str, ...]
    domains: Mapping[str, tuple[float, float]]
    pde_residuals: tuple[AutodiffConstraint, ...]
    conditions: tuple[AutodiffConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("autodiff problem names must be non-empty")
        if not self.variables or any(not name for name in self.variables):
            raise ValueError("autodiff problems require named variables")
        if len(self.variables) != len(set(self.variables)):
            raise ValueError("autodiff problem variables must be unique")
        normalized_domains = dict(self.domains)
        if set(normalized_domains) != set(self.variables):
            raise ValueError("domains must contain exactly the declared variables")
        for variable, (lower, upper) in normalized_domains.items():
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(
                    f"invalid domain for {variable}: bounds must be finite and increasing"
                )
        constraints = self.pde_residuals + self.conditions
        if not constraints:
            raise ValueError("an autodiff problem must contain at least one constraint")
        for constraint in constraints:
            if not isinstance(constraint, AutodiffConstraint):
                raise TypeError("autodiff problem constraints must be AutodiffConstraint objects")
            unknown = set(constraint.fixed_coordinates) - set(self.variables)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"constraint {constraint.name} fixes unknown variables: {names}")
            for variable, value in constraint.fixed_coordinates.items():
                lower, upper = normalized_domains[variable]
                if not lower <= value <= upper:
                    raise ValueError(
                        f"constraint {constraint.name} fixes {variable} outside its domain"
                    )
        object.__setattr__(self, "domains", MappingProxyType(normalized_domains))

    @property
    def constraints(self) -> tuple[AutodiffConstraint, ...]:
        """Return PDE and condition obligations in stable order."""

        return self.pde_residuals + self.conditions


class AutodiffEvaluation:
    """Field, coordinate, and derivative access inside a residual operator."""

    def __init__(self, torch_module, variables, points, fields) -> None:
        self._torch = torch_module
        self._variables = variables
        self._points = points
        self._fields = fields

    def coordinate(self, name: str):
        """Return one coordinate column."""

        try:
            index = self._variables.index(name)
        except ValueError as error:
            raise KeyError(f"unknown coordinate: {name}") from error
        return self._points[:, index : index + 1]

    def field(self, name: str):
        """Return one candidate field column."""

        try:
            return self._fields[name]
        except KeyError as error:
            raise KeyError(f"unknown candidate field: {name}") from error

    def derivative(self, field_name: str, variable: str, *, order: int = 1):
        """Differentiate a field with respect to a coordinate using autograd."""

        if not isinstance(order, int) or order < 1:
            raise ValueError("derivative order must be a positive integer")
        try:
            variable_index = self._variables.index(variable)
        except ValueError as error:
            raise KeyError(f"unknown coordinate: {variable}") from error

        derivative = self.field(field_name)
        for _ in range(order):
            if not derivative.requires_grad:
                return self._torch.zeros_like(derivative)
            gradient = self._torch.autograd.grad(
                derivative,
                self._points,
                grad_outputs=self._torch.ones_like(derivative),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is None:
                return self._torch.zeros_like(derivative)
            derivative = gradient[:, variable_index : variable_index + 1]
        return derivative


@dataclass(frozen=True)
class AutodiffCheckContext:
    """Inputs shared by callable-candidate checkers."""

    problem: AutodiffProblem
    artifact: CallableCandidate
    tolerance: float
    samples_per_axis: int

    @property
    def constraints(self) -> tuple[AutodiffConstraint, ...]:
        """Return obligations in stable order."""

        return self.problem.constraints

    @property
    def obligations(self) -> frozenset[str]:
        """Return all obligation identifiers for conservative aggregation."""

        return frozenset(f"constraint:{index}" for index, _ in enumerate(self.problem.constraints))


class AutodiffResidualChecker:
    """Refute callable fields using deterministic automatic differentiation."""

    name: ClassVar[str] = "autodiff_residual"

    def check(self, context: AutodiffCheckContext) -> CheckResult:
        torch = _load_torch()
        dtype, device = _resolve_tensor_options(torch, context.artifact)
        max_residual = 0.0
        incomplete: dict[str, str] = {}

        for constraint in context.constraints:
            try:
                points = _sample_points(
                    torch,
                    context.problem,
                    constraint,
                    context.samples_per_axis,
                    dtype=dtype,
                    device=device,
                )
                fields = {
                    name: _as_column(torch, function(points), points.shape[0], name)
                    for name, function in context.artifact.fields
                }
                evaluation = AutodiffEvaluation(
                    torch,
                    context.problem.variables,
                    points,
                    fields,
                )
                residual = _as_column(
                    torch,
                    constraint.residual(evaluation),
                    points.shape[0],
                    constraint.name,
                )
                absolute = torch.abs(residual.detach())
                finite = torch.isfinite(absolute)
                if not bool(torch.all(finite)):
                    index = int(torch.nonzero(~finite, as_tuple=False)[0, 0])
                    return CheckResult(
                        witness=_witness(
                            context.problem,
                            constraint,
                            points,
                            index,
                            "undefined",
                            "automatic differentiation produced a non-finite residual",
                        ),
                        max_sampled_residual=float("inf"),
                    )
                value, index_tensor = torch.max(absolute.reshape(-1), dim=0)
                sampled_max = float(value.item())
                index = int(index_tensor.item())
                max_residual = max(max_residual, sampled_max)
                if sampled_max > context.tolerance:
                    return CheckResult(
                        witness=_witness(
                            context.problem,
                            constraint,
                            points,
                            index,
                            sampled_max,
                            "automatic differentiation found a violated obligation",
                        ),
                        max_sampled_residual=max_residual,
                    )
                incomplete[constraint.name] = (
                    "automatic-differentiation samples passed; finite sampling cannot prove "
                    "the obligation"
                )
            except Exception as error:
                incomplete[constraint.name] = (
                    f"automatic-differentiation check raised {type(error).__name__}: {error}"
                )

        return CheckResult(
            incomplete_reasons=incomplete,
            max_sampled_residual=max_residual,
        )


def default_autodiff_checker_registry() -> CheckerRegistry:
    """Return the deterministic built-in callable verification pipeline."""

    return CheckerRegistry((AutodiffResidualChecker(),))


def verify_callable(
    problem: AutodiffProblem,
    artifact: CallableCandidate,
    *,
    tolerance: float = 1e-6,
    samples_per_axis: int = 5,
    checker_registry: CheckerRegistry | None = None,
) -> Report:
    """Verify a callable artifact without treating sampled success as proof."""

    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one")
    context = AutodiffCheckContext(
        problem=problem,
        artifact=artifact,
        tolerance=tolerance,
        samples_per_axis=samples_per_axis,
    )
    return run_checks(context, checker_registry or default_autodiff_checker_registry())


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for callable verification; install pdecert[autodiff]"
        ) from error
    return torch


def _resolve_tensor_options(torch, artifact: CallableCandidate):
    parameter = None
    for _, function in artifact.fields:
        parameters = getattr(function, "parameters", None)
        if callable(parameters):
            parameter = next(iter(parameters()), None)
            if parameter is not None:
                break
    dtype = (
        getattr(torch, artifact.dtype)
        if artifact.dtype is not None
        else parameter.dtype
        if parameter is not None
        else torch.get_default_dtype()
    )
    device = artifact.device or (str(parameter.device) if parameter is not None else "cpu")
    return dtype, device


def _sample_points(torch, problem, constraint, count, *, dtype, device):
    axes = []
    for variable in problem.variables:
        if variable in constraint.fixed_coordinates:
            values = [constraint.fixed_coordinates[variable]]
        else:
            lower, upper = problem.domains[variable]
            base_fractions = [0.113, 0.271, 0.419, 0.613, 0.787, 0.937]
            fractions = (
                base_fractions[:count]
                if count <= len(base_fractions)
                else [(index + 0.5) / count for index in range(count)]
            )
            values = [lower + (upper - lower) * fraction for fraction in fractions]
        axes.append(torch.tensor(values, dtype=dtype, device=device))
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1).requires_grad_(True)


def _as_column(torch, value, point_count: int, name: str):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} returned {type(value).__name__}, not a torch.Tensor")
    if value.ndim == 1 and value.shape[0] == point_count:
        return value.unsqueeze(1)
    if value.ndim == 2 and value.shape == (point_count, 1):
        return value
    raise ValueError(f"{name} must return shape ({point_count},) or ({point_count}, 1)")


def _witness(problem, constraint, points, index, residual, reason):
    return Witness(
        constraint=constraint.name,
        point={
            variable: float(points[index, variable_index].detach().cpu().item())
            for variable_index, variable in enumerate(problem.variables)
        },
        residual=residual,
        reason=reason,
    )
