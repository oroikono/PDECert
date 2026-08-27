"""Evaluate an exact symbolic Burgers wave beside a frozen trained PINN."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import sympy as sp

from experiments.burgers_pinn_fixture import load_frozen_model, validate_integrity_manifest
from pdecert import (
    AutodiffConstraint,
    AutodiffProblem,
    CallableCandidate,
    Constraint,
    EvaluationLane,
    LaneVerificationOptions,
    MatchedCase,
    Problem,
    SymbolicCandidate,
    verify_matched_case,
)


DEFAULT_FIXTURE = Path("benchmarks/matched/burgers-classical-01/pinn.json")
DEFAULT_INTEGRITY = Path("benchmarks/matched/burgers-classical-01/integrity.json")
VISCOSITY = 0.1
WAVE_SPEED = 0.5


def build_case(fixture: str | Path = DEFAULT_FIXTURE) -> tuple[MatchedCase, dict[str, object]]:
    """Build the matched classical problem without sharing candidate implementations."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PDECert with the 'autodiff' extra") from error

    model, manifest = load_frozen_model(fixture)
    x, t = sp.symbols("x t", real=True)
    viscosity = sp.Rational(1, 10)
    wave_speed = sp.Rational(1, 2)
    symbolic_field = wave_speed - sp.tanh((x - wave_speed * t) / (2 * viscosity))
    symbolic_problem = Problem(
        "symbolic viscous Burgers traveling wave",
        (x, t),
        {x: (-1.0, 1.0), t: (0.0, 1.0)},
        (
            Constraint(
                "Burgers PDE",
                sp.diff(symbolic_field, t)
                + symbolic_field * sp.diff(symbolic_field, x)
                - viscosity * sp.diff(symbolic_field, x, 2),
            ),
        ),
        (
            Constraint(
                "initial condition",
                symbolic_field.subs(t, 0) - (wave_speed - sp.tanh(x / (2 * viscosity))),
            ),
            Constraint(
                "left boundary",
                symbolic_field.subs(x, -1)
                - (wave_speed - sp.tanh((-1 - wave_speed * t) / (2 * viscosity))),
            ),
            Constraint(
                "right boundary",
                symbolic_field.subs(x, 1)
                - (wave_speed - sp.tanh((1 - wave_speed * t) / (2 * viscosity))),
            ),
        ),
    )

    def exact_trace(evaluation, x_value):
        return WAVE_SPEED - torch.tanh(
            (x_value - WAVE_SPEED * evaluation.coordinate("t")) / (2.0 * VISCOSITY)
        )

    callable_problem = AutodiffProblem(
        "trained callable viscous Burgers traveling wave",
        ("x", "t"),
        {"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        (
            AutodiffConstraint(
                "Burgers PDE",
                lambda value: value.derivative("u", "t")
                + value.field("u") * value.derivative("u", "x")
                - VISCOSITY * value.derivative("u", "x", order=2),
            ),
        ),
        (
            AutodiffConstraint(
                "initial condition",
                lambda value: value.field("u")
                - (WAVE_SPEED - torch.tanh(value.coordinate("x") / (2.0 * VISCOSITY))),
                {"t": 0.0},
            ),
            AutodiffConstraint(
                "left boundary",
                lambda value: value.field("u") - exact_trace(value, -1.0),
                {"x": -1.0},
            ),
            AutodiffConstraint(
                "right boundary",
                lambda value: value.field("u") - exact_trace(value, 1.0),
                {"x": 1.0},
            ),
        ),
    )
    matched = MatchedCase(
        "burgers-traveling-wave-classical-01",
        ("x", "t"),
        ("u",),
        "classical strong solution",
        (
            EvaluationLane(
                "symbolic-exact",
                symbolic_problem,
                SymbolicCandidate.from_expressions({"u": symbolic_field}),
            ),
            EvaluationLane(
                "trained-pinn",
                callable_problem,
                CallableCandidate.from_mapping({"u": model}, dtype="float64", device="cpu"),
            ),
        ),
    )
    return matched, manifest


def run(
    fixture: str | Path = DEFAULT_FIXTURE,
    integrity: str | Path = DEFAULT_INTEGRITY,
) -> dict[str, object]:
    """Evaluate both lanes while retaining their different evidence strength."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PDECert with the 'autodiff' extra") from error

    fixture_path = Path(fixture)
    integrity_record = validate_integrity_manifest(fixture_path, integrity)
    case, manifest = build_case(fixture_path)
    report = verify_matched_case(
        case,
        options={
            "symbolic-exact": LaneVerificationOptions(
                tolerance=1e-9,
                samples_per_axis=7,
                symbolic_timeout=2.0,
            ),
            "trained-pinn": LaneVerificationOptions(tolerance=1e-3, samples_per_axis=7),
        },
    )
    return {
        "suite": "trained-burgers-pair-v1",
        "scope": "Classical strong-form obligations on x in [-1, 1], t in [0, 1].",
        "evidence_note": (
            "Exact evidence applies only to the symbolic lane. Callable sampling can refute "
            "the frozen PINN or remain inconclusive; it cannot prove a continuous-domain claim."
        ),
        "unsupported": [
            "weak or entropy-solution semantics",
            "discontinuous inviscid shocks",
            "solution-error certification from residual size",
            "architectures other than the declared dense tanh network",
        ],
        "fixture": {
            "artifact_id": manifest["artifact_id"],
            "path": str(fixture_path),
            "sha256": integrity_record["artifact_sha256"],
            "configuration_sha256": integrity_record["configuration_sha256"],
            "source_files_sha256": integrity_record["source_files_sha256"],
            "weights_sha256": manifest["weights_sha256"],
            "training": manifest["training"],
        },
        "torch_version": torch.__version__,
        "matched_report": report.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    payload = run(arguments.fixture, arguments.integrity)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered)
        print(f"Wrote trained Burgers pair result to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
