"""Build deterministic, mechanism-isolating seed records for the PDE Failure Atlas."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from experiments.adversarial_heat import Case, build_cases
from pdecert import (
    Constraint,
    Problem,
    VerificationCase,
    case_to_dict,
    output_sha256,
)


_GENERATED_AT = "2026-08-24T14:30:00+00:00"
_REVISION = "f5a45d8f78e29608bee20e97b75fc9f29ced51f9"
_SOURCE_URL = (
    f"https://github.com/oroikono/PDECert/blob/{_REVISION}/experiments/adversarial_heat.py"
)
_SELECTION = (
    ("synthetic-heat-exact-control", "exact_heat_solution"),
    ("synthetic-heat-pde-only-boundary", "pde_only_boundary_trap"),
    ("synthetic-heat-fixed-grid-alias", "fixed_grid_alias"),
    ("synthetic-heat-hidden-singularity", "hidden_singularity"),
    ("synthetic-heat-single-parameter", "single_parameter_trap"),
    ("synthetic-heat-below-tolerance", "below_numeric_tolerance"),
)


@dataclass(frozen=True)
class Bundle:
    """The three payloads stored in one modular atlas record."""

    metadata: dict[str, object]
    case: dict[str, object]
    raw_output: str


def _serializable_case(case: Case) -> dict[str, object]:
    pde_source = (
        "D(u, t) - k*D(u, x, 2)" if case.name == "single_parameter_trap" else "D(u, t) - D(u, x, 2)"
    )
    condition_sources = (
        "At(u, t, 0) - sin(pi*x)",
        "At(u, x, 0)",
        "At(u, x, 1)",
    )
    problem = Problem(
        name=case.problem.name,
        variables=case.problem.variables,
        domains=dict(case.problem.domains),
        pde_residuals=(
            Constraint(
                case.problem.pde_residuals[0].name,
                case.problem.pde_residuals[0].residual,
                pde_source,
            ),
        ),
        conditions=tuple(
            Constraint(constraint.name, constraint.residual, source)
            for constraint, source in zip(
                case.problem.conditions,
                condition_sources,
                strict=True,
            )
        ),
        parameter_assumptions=dict(case.problem.parameter_assumptions),
    )
    verification_case = VerificationCase(problem, (case.candidate,), ("u",))
    return case_to_dict(verification_case)


def build_bundles() -> dict[str, Bundle]:
    """Return all seed bundles keyed by their stable record identifiers."""

    cases = {case.name: case for case in build_cases()}
    bundles: dict[str, Bundle] = {}
    for record_id, case_name in _SELECTION:
        case = cases[case_name]
        raw_output = f"u(x, t) = {case.candidate}\n"
        metadata = {
            "annotation": {
                "annotators": [],
                "failure_modes": [],
                "rationale": None,
                "status": "pending",
                "verdict": None,
            },
            "id": record_id,
            "origin": {
                "generated_at": _GENERATED_AT,
                "identifier": "experiments.adversarial_heat.build_cases",
                "input": (
                    "Constructed adversarial candidate for the fully stated heat "
                    f"problem. {case.explanation}"
                ),
                "kind": "synthetic",
                "license": "MIT",
                "producer": "PDECert",
                "revision": _REVISION,
                "source_url": _SOURCE_URL,
                "version": "0.1.0",
            },
            "output_sha256": output_sha256(raw_output),
        }
        bundles[record_id] = Bundle(
            metadata=metadata,
            case=_serializable_case(case),
            raw_output=raw_output,
        )
    return bundles


def write_bundles(output: Path) -> None:
    """Write deterministic bundles without touching unrelated records."""

    output.mkdir(parents=True, exist_ok=True)
    for record_id, bundle in build_bundles().items():
        directory = output / record_id
        directory.mkdir(exist_ok=True)
        (directory / "record.json").write_text(
            json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n"
        )
        (directory / "case.json").write_text(
            json.dumps(bundle.case, indent=2, sort_keys=True) + "\n"
        )
        (directory / "raw-output.txt").write_text(bundle.raw_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus/community/records"),
        help="atlas records directory",
    )
    args = parser.parse_args()
    write_bundles(args.output)


if __name__ == "__main__":
    main()
