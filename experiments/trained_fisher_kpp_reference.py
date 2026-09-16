"""Compare a frozen Fisher--KPP PINN with reference values and every obligation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from itertools import product
from pathlib import Path

from pdecert import (
    AutodiffProblem,
    FrozenCallableError,
    __version__ as pdecert_version,
    compare_reference_fields,
    compile_autodiff_problem,
    frozen_callable_to_dict,
    load_frozen_callable,
    load_template,
    materialize_frozen_callable,
    validate_frozen_callable_integrity,
    verify_callable,
)
from pdecert.source_receipts import capture_source_receipt, require_active_source_root

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "fisher-kpp-classical-01"
FIXTURE = Path("benchmarks/matched/fisher-kpp-classical-01/pinn.json")
INTEGRITY = FIXTURE.with_name("integrity.json")
TEMPLATE = FIXTURE.with_name("template.json")
EXPECTED_TEMPLATE_SHA256 = "8d6248c31072d4bc7e109f697ce2637c00420683a68edab7ae99624bc1f6d4a6"
# Bind the original 18-source record as a whole, including its exact inventory.
EXPECTED_INTEGRITY_SHA256 = "cbc76980b4d1ae431bdb438e6b071bde766af83546faf69cf5635ecbe44103d4"
RUNNER_PATHS = (
    "experiments/__init__.py",
    "experiments/trained_fisher_kpp_reference.py",
    "schema/reference-comparison-v1.schema.json",
    "schema/source-receipt-v1.schema.json",
)
FRACTIONS = (0.113, 0.271, 0.419, 0.613, 0.787, 0.937)
TOLERANCE = 1e-3
REGIONS = ("interior", "initial", "left_boundary", "right_boundary")
REFERENCE_DESCRIPTION = "u=(1+exp(x/sqrt(6)-5*t/6))**(-2), evaluated with Python math"
REFERENCE_UNCERTAINTY = (
    "Binary64 evaluation of the analytical traveling-front formula; rounding is not "
    "rigorously bounded. This comparison does not prove the reference solves the PDE "
    "and does not transfer symbolic evidence to the trained PINN."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(
    repository_root: str | Path = ROOT,
    *,
    historical_source_root: str | Path | None = None,
):
    """Validate the preserved artifact and historical files without materialization.

    Omitting the historical root retains strict source checks in the repository
    root. Current evaluation supplies an explicit archive of the original bytes.
    """
    root = Path(repository_root).resolve()
    if _sha256(root / INTEGRITY) != EXPECTED_INTEGRITY_SHA256:
        raise FrozenCallableError("historical integrity digest differs from the preserved record")
    integrity = validate_frozen_callable_integrity(
        root / FIXTURE,
        root / INTEGRITY,
        repository_root=root,
        historical_source_root=historical_source_root,
    )
    template_digest = _sha256(root / TEMPLATE)
    if (
        template_digest != EXPECTED_TEMPLATE_SHA256
        or integrity["source_files_sha256"].get(TEMPLATE.as_posix()) != template_digest
    ):
        raise FrozenCallableError("reference formula supports only the bound Fisher--KPP template")
    template = load_template(root / TEMPLATE)
    frozen = load_frozen_callable(root / FIXTURE)
    if frozen.problem_id != CASE_ID:
        raise FrozenCallableError("reference comparison requires the Fisher--KPP problem ID")
    if frozen.input_names != template.variables or frozen.output_names != template.field_names:
        raise FrozenCallableError("frozen field and coordinate names must match the template")
    return template, frozen, integrity


def sample_coordinates(problem, constraint) -> dict[str, list[float]]:
    """Use the six documented autodiff fractions, preserving x-outer row order."""
    axes = []
    for name in problem.variables:
        if name in constraint.fixed_coordinates:
            axes.append([constraint.fixed_coordinates[name]])
        else:
            lower, upper = problem.domains[name]
            axes.append([lower + (upper - lower) * fraction for fraction in FRACTIONS])
    points = list(product(*axes))
    return {
        name: [point[index] for point in points] for index, name in enumerate(problem.variables)
    }


def _provenance(repository_root: str | Path, integrity) -> dict[str, object]:
    """Bind current package files separately from the preserved historical record."""
    root = Path(repository_root).resolve()
    require_active_source_root(root, (__file__,))
    return {
        "integrity": integrity,
        "active_inputs": {
            name: {"path": path.as_posix(), "sha256": _sha256(root / path)}
            for name, path in (
                ("fixture", FIXTURE),
                ("integrity", INTEGRITY),
                ("template", TEMPLATE),
            )
        },
        "current_evaluator": capture_source_receipt(root, runner_paths=RUNNER_PATHS),
    }


def inspect_inputs(
    repository_root: str | Path = ROOT, *, historical_source_root: str | Path
) -> dict[str, object]:
    """Check historical and current source identity without importing PyTorch."""
    _, _, integrity = load_inputs(repository_root, historical_source_root=historical_source_root)
    return _provenance(repository_root, integrity)


def _evaluate(template, frozen, integrity) -> dict[str, object]:
    """Compute the existing reference and obligation diagnostics with current code."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PDECert with the 'autodiff' extra") from error
    import sympy

    candidate = materialize_frozen_callable(frozen)
    model = dict(candidate.fields)["u"]
    problem = compile_autodiff_problem(template)
    payload = frozen_callable_to_dict(frozen)
    obligations = []
    for index, constraint in enumerate(problem.constraints):
        coordinates = sample_coordinates(problem, constraint)
        points = torch.tensor(
            list(zip(*(coordinates[name] for name in problem.variables))),
            dtype=torch.float64,
            device="cpu",
        )
        with torch.no_grad():
            predictions = model(points).reshape(-1).cpu().tolist()
        references = [
            (1 + math.exp(x / math.sqrt(6) - 5 * t / 6)) ** -2
            for x, t in zip(coordinates["x"], coordinates["t"])
        ]
        samples = {
            "coordinates": coordinates,
            "candidate": {"u": predictions},
            "reference": {"u": references},
        }
        comparison = compare_reference_fields(
            **samples,
            problem_id=CASE_ID,
            candidate_id=payload["artifact_id"],
            reference_id="fisher-kpp-traveling-front-formula",
            reference_description=REFERENCE_DESCRIPTION,
            reference_uncertainty=REFERENCE_UNCERTAINTY,
            sampling_description=(
                f"{REGIONS[index]}: six fixed fractions {FRACTIONS} of each unfixed "
                f"coordinate interval; fixed coordinates {dict(constraint.fixed_coordinates)}; "
                "x outer, t inner; equal weight per row. No refinement or convergence claim."
            ),
        )
        # A full verification may stop at the first failure. Run each obligation
        # alone so an interior violation cannot hide initial/boundary diagnostics.
        single = AutodiffProblem(
            name=problem.name,
            variables=problem.variables,
            domains=problem.domains,
            pde_residuals=(constraint,) if index == 0 else (),
            conditions=() if index == 0 else (constraint,),
        )
        diagnostic = verify_callable(
            single, candidate, tolerance=TOLERANCE, samples_per_axis=len(FRACTIONS)
        )
        obligations.append(
            {
                "source_obligation_id": f"constraint:{index}",
                "name": constraint.name,
                "region": REGIONS[index],
                "samples": samples,
                "reference_comparison": comparison,
                "diagnostic_report": diagnostic.to_dict(),
            }
        )
    return {
        "problem_id": CASE_ID,
        "solution_semantics": template.solution_semantics,
        "evidence_note": (
            "Reference-field errors and PDE/trace residuals are separate empirical quantities. "
            "No combined verdict or solution-error guarantee is produced. Native diagnostic "
            "obligation IDs are local to each single-obligation run; source_obligation_id "
            "identifies the original template obligation."
        ),
        "evaluation": {
            "dtype": "float64",
            "device": "cpu",
            "tolerance": TOLERANCE,
            "samples_per_axis": len(FRACTIONS),
            "fractions": list(FRACTIONS),
            "model_mode": "eval",
            "training_performed": False,
        },
        "fixture": {
            "artifact_id": payload["artifact_id"],
            "artifact_sha256": integrity["artifact_sha256"],
            "configuration_sha256": integrity["configuration_sha256"],
            "weights_sha256": integrity["weights_sha256"],
            "training": payload["training"],
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pdecert_version": pdecert_version,
            "sympy_version": sympy.__version__,
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "obligations": obligations,
    }


def run(
    repository_root: str | Path = ROOT, *, historical_source_root: str | Path
) -> dict[str, object]:
    """Evaluate preserved weights with separately bound current sources, without retraining."""
    template, frozen, integrity = load_inputs(
        repository_root, historical_source_root=historical_source_root
    )
    provenance = _provenance(repository_root, integrity)
    result = _evaluate(template, frozen, integrity)
    if inspect_inputs(repository_root, historical_source_root=historical_source_root) != provenance:
        raise FrozenCallableError("inputs or evaluator sources changed during the operation")
    return {
        "suite": "trained-fisher-kpp-reference-v2",
        "integrity_scope": "content_identity_only",
        "historical_replay": False,
        **provenance,
        **result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--historical-source-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run(arguments.repository_root, historical_source_root=arguments.historical_source_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
