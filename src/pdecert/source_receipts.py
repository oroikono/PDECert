"""Content-only receipts for the active PDECert source tree.

Historical training source hashes belong to frozen integrity records. This
separate contract binds current evaluator files; it never imports archived code.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from .frozen_callable import (
    FrozenCallableError,
    _digest,
    _exact_keys,
    _file_sha256,
    _object,
    _safe_repository_file,
    _validate_json_value,
)


SOURCE_RECEIPT_VERSION = 1
SOURCE_RECEIPT_SCOPE = "content_identity_only"
MAX_SOURCE_RECEIPT_FILES = 256


def _relative_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise FrozenCallableError("source receipt: expected normalized repository-relative path")
    return _safe_repository_file(root, relative, "source receipt")


def _source_paths(root: Path, runner_paths: Sequence[str]) -> list[str]:
    if isinstance(runner_paths, (str, bytes)) or not runner_paths:
        raise FrozenCallableError("runner_paths: expected a non-empty sequence")
    if any(not isinstance(path, str) for path in runner_paths):
        raise FrozenCallableError("runner_paths: expected strings")
    if len(set(runner_paths)) != len(runner_paths):
        raise FrozenCallableError("runner_paths: duplicate source")
    package = root / "src/pdecert"
    paths = {"pyproject.toml", *runner_paths}
    paths.update(path.relative_to(root).as_posix() for path in package.rglob("*.py"))
    if "src/pdecert/source_receipts.py" not in paths:
        raise FrozenCallableError("source receipt: active package source is missing")
    if len(paths) > MAX_SOURCE_RECEIPT_FILES:
        raise FrozenCallableError("source receipt: source count exceeds 256")
    for relative in paths:
        _relative_file(root, relative)
    return sorted(paths)


def capture_source_receipt(
    repository_root: str | Path,
    *,
    runner_paths: Sequence[str],
) -> dict[str, object]:
    """Hash every package Python source, project metadata and explicit runner source.

    The entire package is bound so adding a new implementation module cannot
    silently omit it. External libraries are outside this file closure and must
    be recorded separately in the evaluation runtime metadata.
    """
    root = Path(repository_root).resolve()
    paths = _source_paths(root, runner_paths)
    return {
        "schema_version": SOURCE_RECEIPT_VERSION,
        "integrity_scope": SOURCE_RECEIPT_SCOPE,
        "runner_paths": sorted(runner_paths),
        "source_files_sha256": {
            relative: _file_sha256(_relative_file(root, relative)) for relative in paths
        },
    }


def validate_source_receipt(
    value: object,
    repository_root: str | Path,
    *,
    runner_paths: Sequence[str],
) -> dict[str, object]:
    """Reject changed, missing, additional or unbound current evaluator files."""
    receipt = _object(value, "source receipt")
    _validate_json_value(receipt, "source receipt")
    _exact_keys(
        receipt,
        {"schema_version", "integrity_scope", "runner_paths", "source_files_sha256"},
        "source receipt",
    )
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise FrozenCallableError("source receipt.schema_version: expected 1")
    if receipt["integrity_scope"] != SOURCE_RECEIPT_SCOPE:
        raise FrozenCallableError("source receipt.integrity_scope: expected content_identity_only")
    expected = capture_source_receipt(repository_root, runner_paths=runner_paths)
    if receipt["runner_paths"] != expected["runner_paths"]:
        raise FrozenCallableError("source receipt.runner_paths: does not bind the required runners")
    sources = _object(receipt["source_files_sha256"], "source receipt.source_files_sha256")
    observed = expected["source_files_sha256"]
    if set(sources) != set(observed):
        raise FrozenCallableError("source receipt: source inventory is incomplete or changed")
    for relative, digest in sources.items():
        _digest(digest, f"source receipt.{relative}")
        if digest != observed[relative]:
            raise FrozenCallableError(f"source receipt.{relative}: digest mismatch")
    return receipt


def require_active_source_root(repository_root: str | Path, runner_files: Sequence[str]) -> None:
    """Reject claiming a checkout different from the imported package/runner files.

    This checks origins, not in-memory code attestation or protection against
    monkeypatching. Call only with trusted runner paths, never artifact code.
    """
    root = Path(repository_root).resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "pdecert" and not name.startswith("pdecert."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            raise FrozenCallableError(f"active evaluator module {name}: missing source origin")
        relative = "src/" + name.replace(".", "/")
        if hasattr(module, "__path__"):
            relative += "/__init__.py"
        else:
            relative += ".py"
        if Path(source).resolve() != _relative_file(root, relative):
            raise FrozenCallableError(
                f"active evaluator module {name}: outside the declared source"
            )
    for source in runner_files:
        resolved = Path(source).resolve()
        if root not in resolved.parents:
            raise FrozenCallableError("active runner: outside the declared source")
