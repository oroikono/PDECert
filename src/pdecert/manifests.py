"""Digest-bound manifests for reproducible PDECert evaluation runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .templates import ProblemTemplate, TemplateError, load_template


RUN_MANIFEST_VERSION = 1
INTEGRITY_SCOPE = "content_identity_only"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
_FIELD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


class RunManifestError(ValueError):
    """Raised when a run manifest or its referenced bundle is invalid."""


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunManifestError(f"{path}: expected a non-empty string")
    return value


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RunManifestError(f"{path}: expected an object")
    if any(not isinstance(key, str) for key in value):
        raise RunManifestError(f"{path}: keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise RunManifestError(f"{path}: missing field(s): {', '.join(missing)}")
    if unknown:
        raise RunManifestError(f"{path}: unknown field(s): {', '.join(unknown)}")


def _strict_json_copy(value: object, path: str) -> object:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise RunManifestError(f"{path}: expected strict JSON data: {error}") from error


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _string_mapping(value: Mapping[str, str], path: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise RunManifestError(f"{path}: expected an object")
    normalized = dict(value)
    if not normalized:
        raise RunManifestError(f"{path}: expected at least one entry")
    if any(not isinstance(key, str) or not key.strip() for key in normalized):
        raise RunManifestError(f"{path}: keys must be non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in normalized.values()):
        raise RunManifestError(f"{path}: values must be non-empty strings")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class FileReference:
    """One bundle-local file bound to its exact SHA-256 digest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = _text(self.path, "file.path")
        if "\\" in path:
            raise RunManifestError("file.path: use portable forward slashes")
        parts = path.split("/")
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or any(_PATH_SEGMENT_PATTERN.fullmatch(part) is None for part in parts)
        ):
            raise RunManifestError("file.path: expected a normalized bundle-relative path")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise RunManifestError("file.sha256: expected a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CandidateReference:
    """Identity and provenance for the exact candidate artifact bytes."""

    artifact_id: str
    kind: str
    field_names: tuple[str, ...]
    file: FileReference
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.artifact_id, "candidate.artifact_id")
        if not isinstance(self.kind, str) or _KIND_PATTERN.fullmatch(self.kind) is None:
            raise RunManifestError("candidate.kind: expected a lowercase artifact identifier")
        names = tuple(self.field_names)
        if not names or len(names) != len(set(names)):
            raise RunManifestError("candidate.field_names: expected unique field names")
        if any(
            not isinstance(name, str) or _FIELD_PATTERN.fullmatch(name) is None for name in names
        ):
            raise RunManifestError("candidate.field_names: expected Python identifiers")
        if not isinstance(self.file, FileReference):
            raise TypeError("candidate.file must be a FileReference")
        object.__setattr__(self, "field_names", names)
        object.__setattr__(
            self,
            "provenance",
            _string_mapping(self.provenance, "candidate.provenance"),
        )


@dataclass(frozen=True)
class EvaluatorReference:
    """Evaluator identity, immutable configuration, and runtime environment."""

    name: str
    version: str
    configuration: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.name, "evaluator.name")
        _text(self.version, "evaluator.version")
        if not isinstance(self.configuration, Mapping):
            raise RunManifestError("evaluator.configuration: expected an object")
        normalized = _strict_json_copy(dict(self.configuration), "evaluator.configuration")
        object.__setattr__(self, "configuration", _freeze_json(normalized))
        object.__setattr__(
            self,
            "environment",
            _string_mapping(self.environment, "evaluator.environment"),
        )


@dataclass(frozen=True)
class RunManifest:
    """Content identity and reproduction inputs for one evaluation run."""

    run_id: str
    problem_id: str
    template: FileReference
    candidate: CandidateReference
    evaluator: EvaluatorReference
    report: FileReference

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _text(self.problem_id, "problem_id")
        if not isinstance(self.template, FileReference):
            raise TypeError("template must be a FileReference")
        if not isinstance(self.candidate, CandidateReference):
            raise TypeError("candidate must be a CandidateReference")
        if not isinstance(self.evaluator, EvaluatorReference):
            raise TypeError("evaluator must be an EvaluatorReference")
        if not isinstance(self.report, FileReference):
            raise TypeError("report must be a FileReference")
        paths = (self.template.path, self.candidate.file.path, self.report.path)
        if len(set(paths)) != len(paths):
            raise RunManifestError("template, candidate, and report must reference distinct files")


def _file_from_dict(value: object, path: str) -> FileReference:
    payload = _mapping(value, path)
    _exact_keys(payload, {"path", "sha256"}, path)
    return FileReference(
        path=_text(payload["path"], f"{path}.path"),
        sha256=_text(payload["sha256"], f"{path}.sha256"),
    )


def manifest_from_dict(value: object) -> RunManifest:
    """Validate and parse a version-1 run manifest."""

    payload = _mapping(value, "$")
    _exact_keys(
        payload,
        {
            "manifest_version",
            "integrity_scope",
            "run_id",
            "problem",
            "candidate",
            "evaluator",
            "report",
        },
        "$",
    )
    version = payload["manifest_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != RUN_MANIFEST_VERSION:
        raise RunManifestError(f"$.manifest_version: expected {RUN_MANIFEST_VERSION}")
    if payload["integrity_scope"] != INTEGRITY_SCOPE:
        raise RunManifestError(f"$.integrity_scope: expected {INTEGRITY_SCOPE!r}")

    problem = _mapping(payload["problem"], "$.problem")
    _exact_keys(problem, {"problem_id", "template"}, "$.problem")
    candidate = _mapping(payload["candidate"], "$.candidate")
    _exact_keys(
        candidate,
        {"artifact_id", "kind", "field_names", "file", "provenance"},
        "$.candidate",
    )
    raw_fields = candidate["field_names"]
    if not isinstance(raw_fields, list):
        raise RunManifestError("$.candidate.field_names: expected a list")
    provenance = _mapping(candidate["provenance"], "$.candidate.provenance")

    evaluator = _mapping(payload["evaluator"], "$.evaluator")
    _exact_keys(
        evaluator,
        {"name", "version", "configuration", "environment"},
        "$.evaluator",
    )
    configuration = _mapping(evaluator["configuration"], "$.evaluator.configuration")
    environment = _mapping(evaluator["environment"], "$.evaluator.environment")

    return RunManifest(
        run_id=_text(payload["run_id"], "$.run_id"),
        problem_id=_text(problem["problem_id"], "$.problem.problem_id"),
        template=_file_from_dict(problem["template"], "$.problem.template"),
        candidate=CandidateReference(
            artifact_id=_text(candidate["artifact_id"], "$.candidate.artifact_id"),
            kind=_text(candidate["kind"], "$.candidate.kind"),
            field_names=tuple(raw_fields),
            file=_file_from_dict(candidate["file"], "$.candidate.file"),
            provenance={key: value for key, value in provenance.items()},
        ),
        evaluator=EvaluatorReference(
            name=_text(evaluator["name"], "$.evaluator.name"),
            version=_text(evaluator["version"], "$.evaluator.version"),
            configuration={key: value for key, value in configuration.items()},
            environment={key: value for key, value in environment.items()},
        ),
        report=_file_from_dict(payload["report"], "$.report"),
    )


def manifest_to_dict(manifest: RunManifest) -> dict[str, object]:
    """Convert a run manifest to deterministic JSON-compatible data."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")

    def file_payload(reference: FileReference) -> dict[str, str]:
        return {"path": reference.path, "sha256": reference.sha256}

    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "integrity_scope": INTEGRITY_SCOPE,
        "run_id": manifest.run_id,
        "problem": {
            "problem_id": manifest.problem_id,
            "template": file_payload(manifest.template),
        },
        "candidate": {
            "artifact_id": manifest.candidate.artifact_id,
            "kind": manifest.candidate.kind,
            "field_names": list(manifest.candidate.field_names),
            "file": file_payload(manifest.candidate.file),
            "provenance": dict(manifest.candidate.provenance),
        },
        "evaluator": {
            "name": manifest.evaluator.name,
            "version": manifest.evaluator.version,
            "configuration": _thaw_json(manifest.evaluator.configuration),
            "environment": dict(manifest.evaluator.environment),
        },
        "report": file_payload(manifest.report),
    }


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load a run manifest without resolving its referenced files."""

    source = Path(path)
    try:
        payload = _parse_strict_json(source.read_text())
    except RunManifestError as error:
        raise RunManifestError(f"{source}: invalid JSON: {error}") from error
    return manifest_from_dict(payload)


def dump_run_manifest(manifest: RunManifest, path: str | Path) -> None:
    """Write a run manifest as deterministic, readable JSON."""

    Path(path).write_text(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n")


def run_manifest_sha256(manifest: RunManifest) -> str:
    """Return the digest of the canonical manifest payload."""

    payload = json.dumps(
        manifest_to_dict(manifest),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _parse_strict_json(source: str) -> object:
    try:
        return json.loads(
            source,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RunManifestError(str(error)) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(root: Path, reference: FileReference) -> Path:
    try:
        root = root.resolve(strict=True)
        path = (root / reference.path).resolve(strict=True)
    except OSError as error:
        raise RunManifestError(f"bundle file is unavailable: {reference.path}: {error}") from error
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RunManifestError(
            f"bundle file escapes the manifest directory: {reference.path}"
        ) from error
    if not path.is_file():
        raise RunManifestError(f"bundle reference is not a file: {reference.path}")
    actual = _sha256_file(path)
    if actual != reference.sha256:
        raise RunManifestError(
            f"bundle digest mismatch for {reference.path}: expected {reference.sha256}, got {actual}"
        )
    return path


def _load_strict_report(path: Path) -> Mapping[str, object]:
    try:
        report = _parse_strict_json(path.read_text())
    except RunManifestError as error:
        raise RunManifestError(f"report is not strict JSON: {path}: {error}") from error
    if not isinstance(report, Mapping):
        raise RunManifestError(f"report must contain a JSON object: {path}")
    return report


def _validate_bundle_root(manifest: RunManifest, root: Path) -> ProblemTemplate:
    template_path = _resolve_file(root, manifest.template)
    _resolve_file(root, manifest.candidate.file)
    report_path = _resolve_file(root, manifest.report)
    try:
        template = load_template(template_path)
    except (OSError, TemplateError) as error:
        raise RunManifestError(f"referenced template is invalid: {error}") from error
    if template.field_names != manifest.candidate.field_names:
        raise RunManifestError("candidate field names do not match the referenced problem template")
    _load_strict_report(report_path)
    return template


def validate_run_bundle(path: str | Path) -> RunManifest:
    """Verify all bundle digests and the template/candidate field contract."""

    source = Path(path)
    manifest = load_run_manifest(source)
    _validate_bundle_root(manifest, source.parent)
    return manifest


def _reference_from_path(root: Path, path: str | Path) -> FileReference:
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    try:
        root_resolved = root.resolve(strict=True)
        source_resolved = source.resolve(strict=True)
        relative = source_resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise RunManifestError(f"bundle input must be a file inside {root}: {path}") from error
    if not source_resolved.is_file():
        raise RunManifestError(f"bundle input is not a file: {path}")
    return FileReference(relative.as_posix(), _sha256_file(source_resolved))


def build_run_manifest(
    *,
    bundle_root: str | Path,
    run_id: str,
    problem_id: str,
    template_path: str | Path,
    candidate_path: str | Path,
    report_path: str | Path,
    artifact_id: str,
    artifact_kind: str,
    field_names: tuple[str, ...],
    provenance: Mapping[str, str],
    evaluator_name: str,
    evaluator_version: str,
    evaluator_configuration: Mapping[str, object],
    environment: Mapping[str, str],
) -> RunManifest:
    """Build and validate a manifest from files inside one bundle root."""

    root = Path(bundle_root)
    manifest = RunManifest(
        run_id=run_id,
        problem_id=problem_id,
        template=_reference_from_path(root, template_path),
        candidate=CandidateReference(
            artifact_id=artifact_id,
            kind=artifact_kind,
            field_names=field_names,
            file=_reference_from_path(root, candidate_path),
            provenance=provenance,
        ),
        evaluator=EvaluatorReference(
            name=evaluator_name,
            version=evaluator_version,
            configuration=evaluator_configuration,
            environment=environment,
        ),
        report=_reference_from_path(root, report_path),
    )
    _validate_bundle_root(manifest, root)
    return manifest
