"""Inspect frozen Fisher--KPP provenance, or explicitly evaluate with current code."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pdecert import LaneVerificationOptions, validate_frozen_callable_integrity, verify_matched_case
from pdecert.frozen_callable import (
    FrozenCallableError,
    _exact_keys,
    _digest,
    _file_sha256,
    _load_json,
    _object,
    _text,
    _validate_json_value,
)
from pdecert.source_receipts import (
    capture_source_receipt,
    require_active_source_root,
    validate_source_receipt,
)

from . import trained_fisher_kpp_pair as historical


RUNNER_PATHS = (
    "experiments/__init__.py",
    "experiments/current_fisher_kpp_pair.py",
    "experiments/trained_fisher_kpp_pair.py",
    "schema/source-receipt-v1.schema.json",
)
CONFIGURATION = {
    "symbolic-qwen3": {"tolerance": 1e-9, "samples_per_axis": 6, "symbolic_timeout": 2.0},
    "trained-pinn": {"tolerance": 1e-3, "samples_per_axis": 6, "symbolic_timeout": None},
}
DEFAULT_HISTORY = Path("benchmarks/historical/fisher-kpp-source-v1")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _reference(path: str | Path, root: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if root not in resolved.parents:
        raise FrozenCallableError("active input: path escapes the repository root")
    return {"path": resolved.relative_to(root).as_posix(), "sha256": _file_sha256(resolved)}


def inspect_inputs(
    *,
    historical_source_root: str | Path,
    repository_root: str | Path = ".",
    fixture: str | Path = historical.DEFAULT_FIXTURE,
    integrity: str | Path = historical.DEFAULT_INTEGRITY,
    template: str | Path = historical.DEFAULT_TEMPLATE,
    raw: str | Path = historical.DEFAULT_RAW,
    case: str | Path = historical.DEFAULT_CASE,
    record: str | Path = historical.DEFAULT_RECORD,
) -> dict[str, object]:
    """Validate all historical bytes and bind the active evaluator, without Torch."""
    root = Path(repository_root).resolve()
    require_active_source_root(root, (__file__, historical.__file__))
    integrity_record = validate_frozen_callable_integrity(
        fixture, integrity, root, historical_source_root=historical_source_root
    )
    sources = integrity_record["source_files_sha256"]
    active = {"fixture": fixture, "integrity": integrity}
    for label, path in (("template", template), ("raw", raw), ("case", case), ("record", record)):
        historical._require_bound_input(
            path, label=label, repository_root=root, source_files_sha256=sources
        )
        active[label] = path
    # Structural matching and restricted parsing do not materialize the network.
    historical.build_symbolic_case(template, raw, case, record)
    return {
        "historical_integrity": {
            "file": _reference(integrity, root),
            "source_files_sha256": sources,
            "validation": "all_declared_historical_source_bytes_match",
        },
        "active_inputs": {name: _reference(path, root) for name, path in active.items()},
        "current_evaluator": capture_source_receipt(root, runner_paths=RUNNER_PATHS),
    }


def _runtime() -> dict[str, object]:
    packages = {}
    for name in ("pdecert", "sympy", "mpmath", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": packages,
        "dependency_lock": None,
        "device": "cpu",
        "dtype": "float64",
    }


def _git_context(root: Path) -> dict[str, object]:
    """Best-effort local metadata; exact source-file hashes remain authoritative."""
    try:
        prefix = ["git", "-C", str(root)]
        top = subprocess.check_output(
            prefix + ["rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        if Path(top.decode().strip()).resolve() != root:
            raise ValueError("not this repository")
        revision = subprocess.check_output(
            prefix + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        patch = subprocess.check_output(
            prefix
            + ["diff", "--binary", "HEAD", "--", "src/pdecert", "pyproject.toml", *RUNNER_PATHS],
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            prefix
            + ["ls-files", "--others", "--exclude-standard", "--", "src/pdecert", *RUNNER_PATHS],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"head_revision": None, "tracked_patch_sha256": None, "untracked_paths": None}
    return {
        "head_revision": revision.decode().strip(),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_paths": untracked.decode().splitlines(),
    }


def _validate_runtime_metadata(receipt: dict[str, object]) -> None:
    runtime = _object(receipt["runtime"], "receipt.runtime")
    _exact_keys(
        runtime,
        {"python", "platform", "dependencies", "dependency_lock", "device", "dtype"},
        "receipt.runtime",
    )
    for field in ("python", "platform"):
        _text(runtime[field], f"receipt.runtime.{field}")
    if runtime["device"] != "cpu" or runtime["dtype"] != "float64":
        raise FrozenCallableError("receipt.runtime: expected CPU float64 evaluation")
    if runtime["dependency_lock"] is not None:
        raise FrozenCallableError("receipt.runtime: no dependency lock is defined by this runner")
    dependencies = _object(runtime["dependencies"], "receipt.runtime.dependencies")
    _exact_keys(
        dependencies, {"pdecert", "sympy", "mpmath", "torch"}, "receipt.runtime.dependencies"
    )
    for name, value in dependencies.items():
        if value is not None:
            _text(value, f"receipt.runtime.dependencies.{name}")
    git = _object(receipt["git"], "receipt.git")
    _exact_keys(git, {"head_revision", "tracked_patch_sha256", "untracked_paths"}, "receipt.git")
    if git["head_revision"] is None:
        if git["tracked_patch_sha256"] is not None or git["untracked_paths"] is not None:
            raise FrozenCallableError(
                "receipt.git: unavailable Git context must be explicitly null"
            )
    else:
        head = _text(git["head_revision"], "receipt.git.head_revision")
        if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
            raise FrozenCallableError("receipt.git.head_revision: expected full Git SHA-1")
        _digest(git["tracked_patch_sha256"], "receipt.git.tracked_patch_sha256")
        paths = git["untracked_paths"]
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path for path in paths
        ):
            raise FrozenCallableError("receipt.git.untracked_paths: expected a string list")


def run(*, evaluate: bool = False, **inputs) -> dict[str, object]:
    """Emit a new receipt; numerical evaluation requires an explicit opt-in."""
    if type(evaluate) is not bool:
        raise FrozenCallableError("evaluate: expected a boolean")
    provenance = inspect_inputs(**inputs)
    report = None
    if evaluate:
        matched, _, _ = historical.build_case(
            inputs.get("fixture", historical.DEFAULT_FIXTURE),
            inputs.get("template", historical.DEFAULT_TEMPLATE),
            inputs.get("raw", historical.DEFAULT_RAW),
            inputs.get("case", historical.DEFAULT_CASE),
            inputs.get("record", historical.DEFAULT_RECORD),
        )
        report = verify_matched_case(
            matched,
            options={
                name: LaneVerificationOptions(**options) for name, options in CONFIGURATION.items()
            },
        ).to_dict()
    # A successful receipt must still bind the same inputs and source after work.
    if inspect_inputs(**inputs) != provenance:
        raise FrozenCallableError("inputs or evaluator sources changed during the operation")
    return {
        "receipt_version": 1,
        "suite": "current-fisher-kpp-pair-v1",
        "operation": "current_evaluation" if evaluate else "content_identity_check",
        "integrity_scope": "content_identity_only",
        "historical_replay": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "entrypoint": "experiments.current_fisher_kpp_pair.run",
            "evaluate": evaluate,
            "arguments": {name: str(value) for name, value in inputs.items()},
        },
        **provenance,
        "configuration": copy.deepcopy(CONFIGURATION),
        "runtime": _runtime(),
        "git": _git_context(Path(inputs.get("repository_root", ".")).resolve()),
        "matched_report": report,
        "matched_report_sha256": None if report is None else _canonical_sha256(report),
    }


def validate_receipt(value: object, **inputs) -> dict[str, object]:
    """Check a saved receipt's identity bindings; do not recompute its diagnostics."""
    receipt = _object(value, "receipt")
    _validate_json_value(receipt, "receipt")
    _exact_keys(
        receipt,
        {
            "receipt_version",
            "suite",
            "operation",
            "integrity_scope",
            "historical_replay",
            "created_at",
            "invocation",
            "historical_integrity",
            "active_inputs",
            "current_evaluator",
            "configuration",
            "runtime",
            "git",
            "matched_report",
            "matched_report_sha256",
        },
        "receipt",
    )
    if type(receipt["receipt_version"]) is not int or receipt["receipt_version"] != 1:
        raise FrozenCallableError("receipt.receipt_version: expected 1")
    if receipt["suite"] != "current-fisher-kpp-pair-v1":
        raise FrozenCallableError("receipt.suite: unsupported suite")
    if (
        receipt["integrity_scope"] != "content_identity_only"
        or receipt["historical_replay"] is not False
    ):
        raise FrozenCallableError("receipt: cannot claim historical replay or stronger integrity")
    if receipt["configuration"] != CONFIGURATION:
        raise FrozenCallableError("receipt.configuration: unsupported evaluation configuration")
    try:
        created_at = datetime.fromisoformat(receipt["created_at"])
    except (TypeError, ValueError) as error:
        raise FrozenCallableError("receipt.created_at: expected an ISO timestamp") from error
    if created_at.tzinfo is None:
        raise FrozenCallableError("receipt.created_at: timezone is required")
    invocation = _object(receipt["invocation"], "receipt.invocation")
    _exact_keys(invocation, {"entrypoint", "evaluate", "arguments"}, "receipt.invocation")
    if invocation["entrypoint"] != "experiments.current_fisher_kpp_pair.run":
        raise FrozenCallableError("receipt.invocation: unsupported entrypoint")
    if type(invocation["evaluate"]) is not bool or invocation["evaluate"] != (
        receipt["operation"] == "current_evaluation"
    ):
        raise FrozenCallableError("receipt.invocation: evaluation mode mismatch")
    arguments = _object(invocation["arguments"], "receipt.invocation.arguments")
    if "historical_source_root" not in arguments or any(
        not isinstance(value, str) for value in arguments.values()
    ):
        raise FrozenCallableError("receipt.invocation.arguments: missing root or invalid value")
    _validate_runtime_metadata(receipt)
    observed = inspect_inputs(**inputs)
    validate_source_receipt(
        receipt["current_evaluator"], inputs.get("repository_root", "."), runner_paths=RUNNER_PATHS
    )
    for field in ("historical_integrity", "active_inputs"):
        if receipt[field] != observed[field]:
            raise FrozenCallableError(f"receipt.{field}: content binding mismatch")
    if receipt["operation"] == "content_identity_check":
        if receipt["matched_report"] is not None or receipt["matched_report_sha256"] is not None:
            raise FrozenCallableError(
                "receipt: identity-only inspection cannot claim an evaluation"
            )
    elif receipt["operation"] == "current_evaluation":
        _object(receipt["matched_report"], "receipt.matched_report")
        if _canonical_sha256(receipt["matched_report"]) != receipt["matched_report_sha256"]:
            raise FrozenCallableError("receipt.matched_report_sha256: digest mismatch")
    else:
        raise FrozenCallableError("receipt.operation: unsupported operation")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source-root", required=True, type=Path)
    parser.add_argument("--repository-root", default=Path("."), type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--evaluate", action="store_true", help="run current symbolic/autodiff checks"
    )
    action.add_argument("--validate", type=Path, help="validate a saved receipt without evaluation")
    parser.add_argument("--output", type=Path, help="write a NEW file; existing files are refused")
    arguments = parser.parse_args(argv)
    inputs = {
        "historical_source_root": arguments.historical_source_root,
        "repository_root": arguments.repository_root,
    }
    if arguments.validate is not None:
        payload = validate_receipt(_load_json(arguments.validate), **inputs)
    else:
        payload = run(evaluate=arguments.evaluate, **inputs)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        with arguments.output.open("x") as destination:
            destination.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
