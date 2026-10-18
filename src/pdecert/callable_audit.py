"""Audit every represented obligation of an already-loaded callable model."""

from __future__ import annotations

import math
import platform
from collections.abc import Mapping
from numbers import Real

from .artifacts import CallableCandidate
from .autodiff import AutodiffProblem, _load_torch, _sample_points, verify_callable
from .compiler import compile_autodiff_problem
from .templates import ProblemTemplate, template_to_dict


CALLABLE_AUDIT_VERSION = 1


def audit_callable(
    template: ProblemTemplate,
    artifact: CallableCandidate,
    *,
    candidate_id: str,
    provenance: Mapping[str, str],
    tolerance: float = 1e-6,
    samples_per_axis: int = 5,
    max_sample_points: int = 100_000,
) -> dict[str, object]:
    """Run separate empirical diagnostics for all PDE and condition obligations.

    The caller supplies trusted pointwise fields in physical coordinates, with
    explicit dtype/device and differentiable preprocessing. This function never
    loads checkpoints, trains a model, switches its mode, or changes precision.
    Provenance is caller-declared, not authenticated against the live model.

    Each native report has a local ``constraint:0`` ID; the enclosing
    ``source_obligation_id`` identifies its obligation in the retained template.
    Passing samples stay INCONCLUSIVE. There is no combined audit verdict.
    """
    if not isinstance(template, ProblemTemplate):
        raise TypeError("template must be a ProblemTemplate")
    if not isinstance(artifact, CallableCandidate):
        raise TypeError("artifact must be a CallableCandidate")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("provenance must be a non-empty string mapping")
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in provenance.items()
    ):
        raise ValueError("provenance keys and values must be non-empty strings")
    provenance = dict(provenance)
    if set(artifact.field_names) != set(template.field_names):
        raise ValueError("candidate fields must match the template field names exactly")
    if artifact.dtype is None or artifact.device is None:
        raise ValueError("audit requires explicit candidate dtype and device")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be finite and positive")
    tolerance = float(tolerance)
    for name, value in (
        ("samples_per_axis", samples_per_axis),
        ("max_sample_points", max_sample_points),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    # Compile and admit the entire task before calling any candidate field.
    problem = compile_autodiff_problem(template)
    point_count = sum(
        samples_per_axis ** (len(problem.variables) - len(constraint.fixed_coordinates))
        for constraint in problem.constraints
    )
    if point_count > max_sample_points:
        raise ValueError(
            f"audit requires {point_count} sample points, exceeding max_sample_points="
            f"{max_sample_points}; reduce samples_per_axis or raise the explicit budget"
        )
    torch = _load_torch()
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("audit requires enabled gradients; leave no_grad/inference_mode")
    for name, function in artifact.fields:
        if isinstance(function, torch.nn.Module) and any(
            module.training for module in function.modules()
        ):
            raise ValueError(f"field {name!r} is in training mode; call model.eval() before audit")

    sampled_coordinates = []
    for constraint in problem.constraints:
        points = _sample_points(
            torch,
            problem,
            constraint,
            samples_per_axis,
            dtype=getattr(torch, artifact.dtype),
            device=artifact.device,
        )
        rows = points.detach().cpu().tolist()
        for column, variable in enumerate(problem.variables):
            lower, upper = problem.domains[variable]
            fixed = variable in constraint.fixed_coordinates
            if any(
                not math.isfinite(row[column])
                or not lower <= row[column] <= upper
                or (not fixed and not lower < row[column] < upper)
                for row in rows
            ):
                raise ValueError(
                    f"samples for {constraint.name!r}, coordinate {variable!r}, cannot be "
                    f"represented in the domain with dtype={artifact.dtype}; use a compatible "
                    "dtype or domain"
                )
        sampled_coordinates.append(rows)

    obligations = []
    sources = template.pde_residuals + template.conditions
    for index, (constraint, source, rows) in enumerate(
        zip(problem.constraints, sources, sampled_coordinates)
    ):
        is_pde = index < len(problem.pde_residuals)
        single = AutodiffProblem(
            name=problem.name,
            variables=problem.variables,
            domains=problem.domains,
            pde_residuals=(constraint,) if is_pde else (),
            conditions=() if is_pde else (constraint,),
        )
        report = verify_callable(
            single, artifact, tolerance=tolerance, samples_per_axis=samples_per_axis
        )
        obligations.append(
            {
                "source_obligation_id": f"constraint:{index}",
                "kind": "pde" if is_pde else "condition",
                "name": source.name,
                "expression": source.expression,
                "fixed_coordinates": dict(constraint.fixed_coordinates),
                "sample_points": rows,
                "diagnostic_report": report.to_dict(),
            }
        )

    from . import __version__

    return {
        "audit_version": CALLABLE_AUDIT_VERSION,
        "evidence_scope": "empirical_pointwise_diagnostics",
        "evidence_note": (
            "Each diagnostic_report addresses one template obligation. Its native IDs are local; "
            "source_obligation_id identifies the original template obligation. Passing finite "
            "samples is inconclusive, not proof or a solution-error bound."
        ),
        "template": template_to_dict(template),
        "candidate": {
            "id": candidate_id,
            "field_names": list(artifact.field_names),
            "provenance": provenance,
            "provenance_scope": "caller_declared_not_verified",
        },
        "evaluation": {
            "backend": "torch",
            "dtype": artifact.dtype,
            "device": artifact.device,
            "tolerance": tolerance,
            "samples_per_axis": samples_per_axis,
            "max_sample_points": max_sample_points,
            "total_sample_points": point_count,
            "sampling": "deterministic tensor product; rows follow template variable order",
            "training_performed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pdecert": __version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "obligations": obligations,
    }
