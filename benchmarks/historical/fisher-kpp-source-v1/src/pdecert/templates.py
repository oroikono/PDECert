"""Candidate-free, versioned problem templates and explicit bindings."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import sympy as sp

from .schema import SchemaError, VerificationCase, case_from_dict, case_to_dict


TEMPLATE_VERSION = 1
CLASSICAL_STRONG = "classical_strong"


class TemplateError(ValueError):
    """Raised when a problem template or candidate binding is invalid."""


@dataclass(frozen=True)
class TemplateConstraint:
    """One named, trusted operator expression in a problem template."""

    name: str
    expression: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TemplateError("constraint name must be a non-empty string")
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise TemplateError("constraint expression must be a non-empty string")


@dataclass(frozen=True)
class ProblemTemplate:
    """A trusted PDE specification with no candidate solution attached."""

    name: str
    solution_semantics: str
    variables: tuple[str, ...]
    domains: Mapping[str, tuple[float, float]]
    parameters: Mapping[str, frozenset[str]]
    field_names: tuple[str, ...]
    pde_residuals: tuple[TemplateConstraint, ...]
    conditions: tuple[TemplateConstraint, ...]

    def __post_init__(self) -> None:
        try:
            variables = tuple(self.variables)
            domains = {name: tuple(bounds) for name, bounds in self.domains.items()}
            parameters = {
                name: frozenset(assumptions) for name, assumptions in self.parameters.items()
            }
            field_names = tuple(self.field_names)
            pde_residuals = tuple(self.pde_residuals)
            conditions = tuple(self.conditions)
        except (AttributeError, TypeError) as error:
            raise TemplateError(f"invalid problem template collection: {error}") from error
        if not all(
            isinstance(constraint, TemplateConstraint) for constraint in pde_residuals + conditions
        ):
            raise TemplateError("template constraints must be TemplateConstraint objects")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "domains", MappingProxyType(domains))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(self, "pde_residuals", pde_residuals)
        object.__setattr__(self, "conditions", conditions)
        if self.solution_semantics != CLASSICAL_STRONG:
            raise TemplateError(
                f"solution_semantics must be {CLASSICAL_STRONG!r} in template version 1"
            )
        try:
            case_from_dict(_case_payload(self, {name: "0" for name in self.field_names}))
        except (SchemaError, TypeError, ValueError) as error:
            raise TemplateError(f"invalid problem template: {error}") from error

        sources = tuple(
            constraint.expression for constraint in self.pde_residuals + self.conditions
        )
        missing = sorted(
            name
            for name in self.field_names
            if not any(_contains_name(source, name) for source in sources)
        )
        if missing:
            raise TemplateError(
                "trusted operators do not reference field(s): " + ", ".join(missing)
            )


def _contains_name(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree))


def _constraint_payload(constraints: tuple[TemplateConstraint, ...]) -> list[dict[str, str]]:
    return [
        {"name": constraint.name, "expression": constraint.expression} for constraint in constraints
    ]


def _case_payload(
    template: ProblemTemplate,
    fields: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": template.name,
        "variables": list(template.variables),
        "domains": {name: list(bounds) for name, bounds in template.domains.items()},
        "parameters": {
            name: sorted(assumptions) for name, assumptions in template.parameters.items()
        },
        "fields": dict(fields),
        "pde_residuals": _constraint_payload(template.pde_residuals),
        "conditions": _constraint_payload(template.conditions),
    }


def template_from_dict(value: object) -> ProblemTemplate:
    """Validate and parse one version-1 problem template."""

    if not isinstance(value, Mapping):
        raise TemplateError("$: expected an object")
    required = {
        "template_version",
        "name",
        "solution_semantics",
        "variables",
        "domains",
        "parameters",
        "field_names",
        "pde_residuals",
        "conditions",
    }
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise TemplateError("$: missing field(s): " + ", ".join(missing))
    if unknown:
        raise TemplateError("$: unknown field(s): " + ", ".join(unknown))
    if value["template_version"] != TEMPLATE_VERSION or isinstance(value["template_version"], bool):
        raise TemplateError(f"$.template_version: expected {TEMPLATE_VERSION}")
    if not isinstance(value["variables"], list):
        raise TemplateError("$.variables: expected a list")
    if not isinstance(value["field_names"], list):
        raise TemplateError("$.field_names: expected a list")
    if not isinstance(value["domains"], Mapping):
        raise TemplateError("$.domains: expected an object")
    if not isinstance(value["parameters"], Mapping):
        raise TemplateError("$.parameters: expected an object")

    def constraints(raw: object, path: str) -> tuple[TemplateConstraint, ...]:
        if not isinstance(raw, list):
            raise TemplateError(f"{path}: expected a list")
        parsed: list[TemplateConstraint] = []
        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"
            if not isinstance(item, Mapping):
                raise TemplateError(f"{item_path}: expected an object")
            if set(item) != {"name", "expression"}:
                raise TemplateError(f"{item_path}: expected name and expression only")
            parsed.append(TemplateConstraint(item["name"], item["expression"]))
        return tuple(parsed)

    raw_domains = value["domains"]
    raw_parameters = value["parameters"]
    try:
        domains = {name: tuple(bounds) for name, bounds in raw_domains.items()}
        parameters = {name: frozenset(assumptions) for name, assumptions in raw_parameters.items()}
    except TypeError as error:
        raise TemplateError(f"invalid template collection: {error}") from error
    return ProblemTemplate(
        name=value["name"],
        solution_semantics=value["solution_semantics"],
        variables=tuple(value["variables"]),
        domains=domains,
        parameters=parameters,
        field_names=tuple(value["field_names"]),
        pde_residuals=constraints(value["pde_residuals"], "$.pde_residuals"),
        conditions=constraints(value["conditions"], "$.conditions"),
    )


def template_to_dict(template: ProblemTemplate) -> dict[str, object]:
    """Convert a problem template into deterministic JSON-compatible data."""

    if not isinstance(template, ProblemTemplate):
        raise TypeError("template must be a ProblemTemplate")
    return {
        "template_version": TEMPLATE_VERSION,
        "name": template.name,
        "solution_semantics": template.solution_semantics,
        "variables": list(template.variables),
        "domains": {name: list(bounds) for name, bounds in template.domains.items()},
        "parameters": {
            name: sorted(assumptions) for name, assumptions in template.parameters.items()
        },
        "field_names": list(template.field_names),
        "pde_residuals": _constraint_payload(template.pde_residuals),
        "conditions": _constraint_payload(template.conditions),
    }


def load_template(path: str | Path) -> ProblemTemplate:
    """Load and validate a candidate-free problem template from JSON."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise TemplateError(f"{source}: invalid JSON: {error.msg}") from error
    return template_from_dict(payload)


def dump_template(template: ProblemTemplate, path: str | Path) -> None:
    """Write a problem template as deterministic, readable JSON."""

    Path(path).write_text(json.dumps(template_to_dict(template), indent=2, sort_keys=True) + "\n")


def template_from_case(
    case: VerificationCase,
    *,
    solution_semantics: str = CLASSICAL_STRONG,
) -> ProblemTemplate:
    """Remove the candidate binding from a fully instantiated case."""

    if not isinstance(case, VerificationCase):
        raise TypeError("case must be a VerificationCase")
    payload = case_to_dict(case)
    return template_from_dict(
        {
            "template_version": TEMPLATE_VERSION,
            "name": payload["name"],
            "solution_semantics": solution_semantics,
            "variables": payload["variables"],
            "domains": payload["domains"],
            "parameters": payload["parameters"],
            "field_names": list(payload["fields"]),
            "pde_residuals": payload["pde_residuals"],
            "conditions": payload["conditions"],
        }
    )


def bind_symbolic_candidate(
    template: ProblemTemplate,
    fields: Mapping[str, str | sp.Expr],
) -> VerificationCase:
    """Bind exactly one expression to every declared template field."""

    if not isinstance(template, ProblemTemplate):
        raise TypeError("template must be a ProblemTemplate")
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping")
    missing = sorted(set(template.field_names) - set(fields))
    unknown = sorted(set(fields) - set(template.field_names))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise TemplateError("candidate fields do not match template (" + "; ".join(details) + ")")

    rendered: dict[str, str] = {}
    for name in template.field_names:
        expression = fields[name]
        if isinstance(expression, str):
            rendered[name] = expression
        elif isinstance(expression, sp.Expr):
            rendered[name] = sp.sstr(expression)
        else:
            raise TemplateError(f"candidate field {name!r} must be a string or SymPy expression")
    try:
        return case_from_dict(_case_payload(template, rendered))
    except SchemaError as error:
        raise TemplateError(f"candidate binding failed: {error}") from error
