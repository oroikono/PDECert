"""Evaluate a raw symbolic Fisher--KPP proposal beside a frozen trained PINN."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path

from pdecert import (
    EvaluationLane,
    FrozenCallableError,
    LaneVerificationOptions,
    MatchedCase,
    SymbolicCandidate,
    __version__ as pdecert_version,
    bind_symbolic_candidate,
    compile_autodiff_problem,
    frozen_callable_to_dict,
    load_case,
    load_frozen_callable,
    load_template,
    materialize_frozen_callable,
    template_from_case,
    template_to_dict,
    validate_frozen_callable_integrity,
    verify_matched_case,
)


CASE_ID = "fisher-kpp-classical-01"
DEFAULT_TEMPLATE = Path("benchmarks/matched/fisher-kpp-classical-01/template.json")
DEFAULT_FIXTURE = Path("benchmarks/matched/fisher-kpp-classical-01/pinn.json")
DEFAULT_INTEGRITY = Path("benchmarks/matched/fisher-kpp-classical-01/integrity.json")
DEFAULT_RAW = Path("corpus/community/records/qwen3-fisher-kpp-01/raw-output.txt")
DEFAULT_CASE = Path("corpus/community/records/qwen3-fisher-kpp-01/case.json")
DEFAULT_RECORD = Path("corpus/community/records/qwen3-fisher-kpp-01/record.json")
CALLABLE_TOLERANCE = 1e-3
SAMPLES_PER_AXIS = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_bound_input(
    path: str | Path,
    *,
    label: str,
    repository_root: str | Path,
    source_files_sha256: dict[str, object],
) -> None:
    root = Path(repository_root).resolve()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise FrozenCallableError(f"{label}: path escapes the repository root")
    relative = resolved.relative_to(root).as_posix()
    expected = source_files_sha256.get(relative)
    if not isinstance(expected, str):
        raise FrozenCallableError(f"{label}: active input is not bound by the integrity record")
    if _sha256(resolved) != expected:
        raise FrozenCallableError(
            f"{label}: active input digest does not match the integrity record"
        )


def load_symbolic_proposal(
    raw: str | Path = DEFAULT_RAW,
    case: str | Path = DEFAULT_CASE,
    record: str | Path = DEFAULT_RECORD,
) -> tuple[str, dict[str, object]]:
    """Load the unedited Qwen proposal and bind it to its corpus provenance."""

    raw_path = Path(raw)
    case_path = Path(case)
    record_path = Path(record)
    raw_text = raw_path.read_text()
    case_payload = json.loads(case_path.read_text())
    record_payload = json.loads(record_path.read_text())
    if not isinstance(case_payload, dict) or not isinstance(record_payload, dict):
        raise ValueError("Fisher--KPP corpus case and record must be JSON objects")

    raw_sha256 = _sha256(raw_path)
    if record_payload.get("output_sha256") != raw_sha256:
        raise ValueError("Fisher--KPP raw output does not match its corpus record digest")
    prefix = "FINAL u:"
    if not raw_text.startswith(prefix):
        raise ValueError("Fisher--KPP raw output must begin with 'FINAL u:'")
    expression = raw_text[len(prefix) :].strip()
    if not expression:
        raise ValueError("Fisher--KPP raw output has no candidate expression")

    annotation = record_payload.get("annotation")
    if not isinstance(annotation, dict) or not isinstance(annotation.get("status"), str):
        raise ValueError("Fisher--KPP corpus record has no explicit annotation status")
    provenance = {
        "record_id": record_payload.get("id"),
        "candidate_expression": expression,
        "annotation": annotation,
        "origin": record_payload.get("origin"),
        "files": {
            "raw_output": {"path": str(raw_path), "sha256": raw_sha256},
            "case": {"path": str(case_path), "sha256": _sha256(case_path)},
            "record": {"path": str(record_path), "sha256": _sha256(record_path)},
        },
    }
    return expression, provenance


def build_symbolic_case(
    template: str | Path = DEFAULT_TEMPLATE,
    raw: str | Path = DEFAULT_RAW,
    case: str | Path = DEFAULT_CASE,
    record: str | Path = DEFAULT_RECORD,
):
    """Bind the raw candidate to the trusted candidate-free problem template."""

    problem_template = load_template(template)
    expression, provenance = load_symbolic_proposal(raw, case, record)
    corpus_template = template_from_case(load_case(case))
    if template_to_dict(problem_template) != template_to_dict(corpus_template):
        raise ValueError("Fisher--KPP template does not match the preserved corpus problem")
    return (
        problem_template,
        bind_symbolic_candidate(problem_template, {"u": expression}),
        provenance,
    )


def build_case(
    fixture: str | Path = DEFAULT_FIXTURE,
    template: str | Path = DEFAULT_TEMPLATE,
    raw: str | Path = DEFAULT_RAW,
    case: str | Path = DEFAULT_CASE,
    record: str | Path = DEFAULT_RECORD,
):
    """Build both lanes from one trusted template and two independent candidates."""

    problem_template, symbolic_case, provenance = build_symbolic_case(
        template,
        raw,
        case,
        record,
    )
    frozen = load_frozen_callable(fixture)
    frozen_payload = frozen_callable_to_dict(frozen)
    if frozen_payload["problem_id"] != CASE_ID:
        raise FrozenCallableError(
            f"$.problem_id: expected {CASE_ID!r}, observed {frozen_payload['problem_id']!r}"
        )
    architecture = frozen_payload["architecture"]
    if tuple(architecture["input_names"]) != problem_template.variables:
        raise FrozenCallableError("$.architecture.input_names: do not match template coordinates")
    if tuple(architecture["output_names"]) != problem_template.field_names:
        raise FrozenCallableError("$.architecture.output_names: do not match template fields")

    callable_candidate = materialize_frozen_callable(frozen)
    matched = MatchedCase(
        CASE_ID,
        problem_template.variables,
        problem_template.field_names,
        problem_template.solution_semantics,
        (
            EvaluationLane(
                "symbolic-qwen3",
                symbolic_case.problem,
                SymbolicCandidate.from_expressions(symbolic_case.candidate_fields),
            ),
            EvaluationLane(
                "trained-pinn",
                compile_autodiff_problem(problem_template),
                callable_candidate,
            ),
        ),
    )
    return matched, frozen, provenance


def run(
    fixture: str | Path = DEFAULT_FIXTURE,
    integrity: str | Path = DEFAULT_INTEGRITY,
    template: str | Path = DEFAULT_TEMPLATE,
    raw: str | Path = DEFAULT_RAW,
    case: str | Path = DEFAULT_CASE,
    record: str | Path = DEFAULT_RECORD,
    *,
    repository_root: str | Path = ".",
) -> dict[str, object]:
    """Evaluate both lanes without promoting sampled callable evidence."""

    fixture_path = Path(fixture)
    integrity_record = validate_frozen_callable_integrity(
        fixture_path,
        integrity,
        repository_root=repository_root,
    )
    source_files_sha256 = integrity_record["source_files_sha256"]
    if not isinstance(source_files_sha256, dict):
        raise FrozenCallableError("$.source_files_sha256: expected an object")
    for label, active_path in (
        ("template", template),
        ("raw", raw),
        ("case", case),
        ("record", record),
    ):
        _require_bound_input(
            active_path,
            label=label,
            repository_root=repository_root,
            source_files_sha256=source_files_sha256,
        )
    matched_case, frozen, provenance = build_case(fixture_path, template, raw, case, record)
    frozen_payload = frozen_callable_to_dict(frozen)
    report = verify_matched_case(
        matched_case,
        options={
            "symbolic-qwen3": LaneVerificationOptions(
                tolerance=1e-9,
                samples_per_axis=SAMPLES_PER_AXIS,
                symbolic_timeout=2.0,
            ),
            "trained-pinn": LaneVerificationOptions(
                tolerance=CALLABLE_TOLERANCE,
                samples_per_axis=SAMPLES_PER_AXIS,
            ),
        },
    )
    import sympy
    import torch

    return {
        "suite": "trained-fisher-kpp-pair-v1",
        "scope": "Classical strong-form obligations on x in [-6, 6], t in [0, 2].",
        "evidence_note": (
            "Exact evidence applies only to the raw symbolic candidate after restricted "
            "binding. Callable sampling can refute the frozen PINN or remain inconclusive; "
            "it cannot prove a continuous-domain claim."
        ),
        "evaluation": {
            "callable_tolerance": CALLABLE_TOLERANCE,
            "samples_per_axis": SAMPLES_PER_AXIS,
        },
        "runtime": {
            "pdecert_version": pdecert_version,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "sympy_version": sympy.__version__,
            "torch_version": torch.__version__,
        },
        "unsupported": [
            "weak or generalized solution semantics",
            "traveling-front stability or uniqueness",
            "solution-error certification from residual size",
            "architectures other than the declared dense tanh network",
        ],
        "symbolic_proposal": provenance,
        "fixture": {
            "artifact_id": frozen_payload["artifact_id"],
            "path": str(fixture_path),
            "sha256": integrity_record["artifact_sha256"],
            "configuration_sha256": integrity_record["configuration_sha256"],
            "source_files_sha256": integrity_record["source_files_sha256"],
            "weights_sha256": frozen_payload["weights_sha256"],
            "training": frozen_payload["training"],
        },
        "matched_report": report.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    payload = run(
        arguments.fixture,
        arguments.integrity,
        arguments.template,
        arguments.raw,
        arguments.case,
        arguments.record,
        repository_root=arguments.repository_root,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered)
        print(f"Wrote trained Fisher--KPP pair result to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
