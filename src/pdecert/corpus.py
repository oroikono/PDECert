"""Versioned, provenance-bearing corpus records for generated PDE candidates."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .frozen_callable import (
    FrozenCallableError,
    canonical_frozen_configuration_sha256,
    frozen_callable_to_dict,
    load_frozen_callable,
)
from .schema import SCHEMA_VERSION, case_from_dict
from .templates import TemplateError, bind_symbolic_candidate, template_from_dict, template_to_dict


CORPUS_VERSION = 1
ATLAS_VERSION = 1
CROSS_ARTIFACT_ATLAS_VERSION = 2
CROSS_ARTIFACT_RECORD_VERSION = 1
SYMBOLIC_ARTIFACT_VERSION = 1
CROSS_ARTIFACT_METADATA_MAX_BYTES = 1_000_000
COVERAGE_VERSION = 1
ARTIFACT_TYPES = frozenset(
    {"callable_model", "numerical_field", "solver_program", "symbolic_expression"}
)
ORIGIN_KINDS = frozenset({"open_model", "symbolic_solver", "synthetic"})
FAILURE_MODES = frozenset(
    {
        "boundary_condition",
        "domain_singularity",
        "extraction_error",
        "initial_condition",
        "other",
        "parameter_scope",
        "pde_residual",
        "unsupported_semantics",
    }
)
ANNOTATION_STATUSES = frozenset({"adjudicated", "labeled", "pending"})
VERDICTS = frozenset({"invalid", "unclear", "valid"})
_RECORD_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TAXONOMY_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_BUNDLE_FILES = frozenset({"case.json", "raw-output.txt", "record.json"})
_CROSS_ARTIFACT_TYPES = frozenset({"callable_model", "symbolic_expression"})
_CROSS_ORIGIN_KINDS = ORIGIN_KINDS | {"trained_model"}
_CROSS_RECORD_FIELDS = {
    "annotation",
    "artifact_type",
    "files",
    "id",
    "origin",
    "problem_id",
    "record_version",
}
_FILE_REFERENCE_FIELDS = {"path", "sha256"}
_SYMBOLIC_ARTIFACT_FIELDS = {
    "artifact_id",
    "artifact_kind",
    "artifact_version",
    "fields",
    "problem_id",
    "raw_output_sha256",
}
_FROZEN_INTEGRITY_FIELDS = {
    "artifact_path",
    "artifact_sha256",
    "base_revision",
    "configuration_sha256",
    "schema_version",
    "source_files_sha256",
    "training_run",
    "weights_sha256",
}


class CorpusError(ValueError):
    """Raised when a candidate corpus does not follow the corpus schema."""


def output_sha256(raw_output: str) -> str:
    """Return the content digest stored with one raw generator output."""

    return hashlib.sha256(raw_output.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CorpusError(f"{path}: could not read file: {error}") from error
    return digest.hexdigest()


def _error(path: str, message: str) -> CorpusError:
    return CorpusError(f"{path}: {message}")


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(path, "expected an object")
    return value


def _exact_keys(value: Mapping[str, object], required: set[str], path: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise _error(path, f"missing field(s): {', '.join(missing)}")
    if unknown:
        raise _error(path, f"unknown field(s): {', '.join(unknown)}")


def _text(value: object, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "expected a non-empty string")
    return value


def _digest(value: object, path: str) -> str:
    digest = _text(value, path)
    if _SHA256.fullmatch(digest) is None:
        raise _error(path, "expected a lowercase SHA-256 digest")
    return digest


def _validate_timestamp(value: object, path: str) -> None:
    source = _text(value, path)
    try:
        parsed = datetime.fromisoformat(source.replace("Z", "+00:00"))
    except ValueError as error:
        raise _error(path, "expected an ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise _error(path, "timestamp must include a UTC offset")


def _validate_url(value: object, path: str) -> None:
    source = _text(value, path)
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _error(path, "expected an absolute HTTP(S) URL")


def _validate_origin(
    value: object,
    path: str,
    *,
    allowed_kinds: frozenset[str] = ORIGIN_KINDS,
) -> None:
    origin = _object(value, path)
    fields = {
        "generated_at",
        "identifier",
        "input",
        "kind",
        "license",
        "producer",
        "revision",
        "source_url",
        "version",
    }
    _exact_keys(origin, fields, path)
    kind = _text(origin["kind"], f"{path}.kind")
    if kind not in allowed_kinds:
        raise _error(f"{path}.kind", f"expected one of: {', '.join(sorted(allowed_kinds))}")
    for name in ("identifier", "input", "producer"):
        _text(origin[name], f"{path}.{name}")
    for name in ("license", "revision", "version"):
        _text(origin[name], f"{path}.{name}", nullable=True)
    _validate_url(origin["source_url"], f"{path}.source_url")
    _validate_timestamp(origin["generated_at"], f"{path}.generated_at")


def _validate_annotation(value: object, path: str) -> None:
    annotation = _object(value, path)
    fields = {"annotators", "failure_modes", "rationale", "status", "verdict"}
    _exact_keys(annotation, fields, path)
    status = _text(annotation["status"], f"{path}.status")
    if status not in ANNOTATION_STATUSES:
        raise _error(
            f"{path}.status",
            f"expected one of: {', '.join(sorted(ANNOTATION_STATUSES))}",
        )
    verdict = _text(annotation["verdict"], f"{path}.verdict", nullable=True)
    if verdict is not None and verdict not in VERDICTS:
        raise _error(f"{path}.verdict", f"expected one of: {', '.join(sorted(VERDICTS))}")
    rationale = _text(annotation["rationale"], f"{path}.rationale", nullable=True)

    annotators = annotation["annotators"]
    if not isinstance(annotators, list) or any(
        not isinstance(item, str) or not item.strip() for item in annotators
    ):
        raise _error(f"{path}.annotators", "expected a list of non-empty strings")
    if len(set(annotators)) != len(annotators):
        raise _error(f"{path}.annotators", "annotators must be unique")

    failure_modes = annotation["failure_modes"]
    if not isinstance(failure_modes, list):
        raise _error(f"{path}.failure_modes", "expected a list")
    unknown_modes = (
        set(failure_modes) - FAILURE_MODES
        if all(isinstance(item, str) for item in failure_modes)
        else {"non-string value"}
    )
    if unknown_modes:
        raise _error(
            f"{path}.failure_modes",
            f"unsupported failure mode(s): {', '.join(sorted(unknown_modes))}",
        )
    if len(set(failure_modes)) != len(failure_modes):
        raise _error(f"{path}.failure_modes", "failure modes must be unique")

    if status == "pending":
        if verdict is not None or rationale is not None or annotators or failure_modes:
            raise _error(path, "pending annotations must not contain a label")
        return
    if verdict is None or rationale is None or not annotators:
        raise _error(path, "completed annotations require verdict, rationale, and annotator")
    if status == "adjudicated" and len(annotators) < 2:
        raise _error(path, "adjudicated annotations require at least two annotators")
    if verdict == "invalid" and not failure_modes:
        raise _error(path, "invalid verdicts require at least one failure mode")
    if verdict != "invalid" and failure_modes:
        raise _error(path, "failure modes are only valid for an invalid verdict")


def _validate_record(value: object, path: str) -> None:
    record = _object(value, path)
    fields = {"annotation", "case", "id", "origin", "output_sha256", "raw_output"}
    _exact_keys(record, fields, path)
    record_id = _text(record["id"], f"{path}.id")
    if not _RECORD_ID.fullmatch(record_id):
        raise _error(f"{path}.id", "expected a lowercase corpus identifier")
    raw_output = _text(record["raw_output"], f"{path}.raw_output")
    digest = _text(record["output_sha256"], f"{path}.output_sha256")
    if digest != output_sha256(raw_output):
        raise _error(f"{path}.output_sha256", "does not match raw_output")
    case = _object(record["case"], f"{path}.case")
    if case.get("schema_version") != SCHEMA_VERSION:
        raise _error(f"{path}.case.schema_version", f"expected {SCHEMA_VERSION}")
    try:
        case_from_dict(case)
    except ValueError as error:
        raise _error(f"{path}.case", str(error)) from error
    _validate_origin(record["origin"], f"{path}.origin")
    _validate_annotation(record["annotation"], f"{path}.annotation")


def validate_corpus(value: object) -> None:
    """Validate a complete corpus document or raise :class:`CorpusError`."""

    corpus = _object(value, "$")
    _exact_keys(corpus, {"corpus_version", "description", "name", "records"}, "$")
    if isinstance(corpus["corpus_version"], bool) or corpus["corpus_version"] != CORPUS_VERSION:
        raise _error("$.corpus_version", f"expected {CORPUS_VERSION}")
    _text(corpus["name"], "$.name")
    _text(corpus["description"], "$.description")
    records = corpus["records"]
    if not isinstance(records, list):
        raise _error("$.records", "expected a list")
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        path = f"$.records[{index}]"
        _validate_record(record, path)
        record_id = record["id"]
        if record_id in identifiers:
            raise _error(path, f"duplicate record id: {record_id}")
        identifiers.add(record_id)


def validate_atlas_coverage(value: object, record_ids: set[str]) -> None:
    """Validate explicit coverage metadata against an Atlas record set."""

    coverage = _object(value, "$")
    _exact_keys(coverage, {"coverage_version", "records"}, "$")
    version = coverage["coverage_version"]
    if isinstance(version, bool) or version != COVERAGE_VERSION:
        raise _error("$.coverage_version", f"expected {COVERAGE_VERSION}")

    entries = _object(coverage["records"], "$.records")
    if any(not isinstance(record_id, str) for record_id in entries):
        raise _error("$.records", "record ids must be strings")
    actual_ids = set(entries)
    missing = sorted(record_ids - actual_ids)
    unknown = sorted(actual_ids - record_ids)
    if missing:
        raise _error("$.records", f"missing record id(s): {', '.join(missing)}")
    if unknown:
        raise _error("$.records", f"unknown record id(s): {', '.join(unknown)}")

    for record_id in sorted(record_ids):
        path = f"$.records.{record_id}"
        entry = _object(entries[record_id], path)
        _exact_keys(entry, {"artifact_type", "pde_families", "spatial_dimension"}, path)

        artifact_type = _text(entry["artifact_type"], f"{path}.artifact_type")
        if artifact_type not in ARTIFACT_TYPES:
            raise _error(
                f"{path}.artifact_type",
                f"expected one of: {', '.join(sorted(ARTIFACT_TYPES))}",
            )

        families = entry["pde_families"]
        if not isinstance(families, list) or not families:
            raise _error(f"{path}.pde_families", "expected a non-empty list")
        if any(
            not isinstance(family, str) or not _TAXONOMY_SLUG.fullmatch(family)
            for family in families
        ):
            raise _error(
                f"{path}.pde_families",
                "expected lowercase taxonomy slugs",
            )
        if len(set(families)) != len(families):
            raise _error(f"{path}.pde_families", "PDE families must be unique")

        spatial_dimension = entry["spatial_dimension"]
        if (
            isinstance(spatial_dimension, bool)
            or not isinstance(spatial_dimension, int)
            or spatial_dimension < 1
        ):
            raise _error(f"{path}.spatial_dimension", "expected a positive integer")


def _load_json(source: Path) -> object:
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise CorpusError(f"{source}: invalid JSON: {error.msg}") from error
    return payload


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise CorpusError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise CorpusError(f"non-finite JSON number is not supported: {value}")


def _load_cross_artifact_json(source: Path) -> object:
    try:
        if source.stat().st_size > CROSS_ARTIFACT_METADATA_MAX_BYTES:
            raise CorpusError(
                f"{source}: metadata exceeds the {CROSS_ARTIFACT_METADATA_MAX_BYTES}-byte limit"
            )
        text = source.read_text(encoding="utf-8")
    except CorpusError:
        raise
    except (OSError, UnicodeError) as error:
        raise CorpusError(f"{source}: could not read file: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise CorpusError(f"{source}: invalid JSON: {error.msg}") from error


def load_corpus(path: str | Path) -> dict[str, Any]:
    """Load and validate one corpus JSON document."""

    source = Path(path)
    payload = _load_json(source)
    validate_corpus(payload)
    return payload


def load_atlas_coverage(path: str | Path, record_ids: set[str]) -> dict[str, Any]:
    """Load and validate one Atlas coverage taxonomy document."""

    source = Path(path) / "coverage.json"
    if source.is_symlink() or not source.is_file():
        raise CorpusError(f"{source}: expected a regular file")
    payload = _load_json(source)
    validate_atlas_coverage(payload, record_ids)
    return payload


def load_record_bundle(path: str | Path) -> dict[str, Any]:
    """Load one modular atlas record and reconstruct its corpus representation."""

    source = Path(path)
    entries = {entry.name: entry for entry in source.iterdir()}
    missing = sorted(_BUNDLE_FILES - set(entries))
    unknown = sorted(set(entries) - _BUNDLE_FILES)
    if missing:
        raise CorpusError(f"{source}: missing bundle file(s): {', '.join(missing)}")
    if unknown:
        raise CorpusError(f"{source}: unexpected bundle file(s): {', '.join(unknown)}")
    for entry in entries.values():
        if entry.is_symlink() or not entry.is_file():
            raise CorpusError(f"{entry}: bundle entries must be regular files")

    metadata = _object(_load_json(source / "record.json"), "$")
    _exact_keys(
        metadata,
        {"annotation", "id", "origin", "output_sha256"},
        "$",
    )
    case = _object(_load_json(source / "case.json"), "$.case")
    raw_path = source / "raw-output.txt"
    try:
        raw_output = raw_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusError(f"{raw_path}: raw output must be valid UTF-8") from error
    record = {
        "annotation": metadata["annotation"],
        "case": dict(case),
        "id": metadata["id"],
        "origin": metadata["origin"],
        "output_sha256": metadata["output_sha256"],
        "raw_output": raw_output,
    }
    _validate_record(record, "$")
    if source.name != record["id"]:
        raise CorpusError(f"{source}: directory name must match record id {record['id']!r}")
    return record


def load_atlas(path: str | Path) -> dict[str, Any]:
    """Load a modular atlas directory as a validated corpus document."""

    source = Path(path)
    manifest = _object(_load_json(source / "atlas.json"), "$")
    _exact_keys(manifest, {"atlas_version", "description", "name"}, "$")
    version = manifest["atlas_version"]
    if isinstance(version, bool) or version != ATLAS_VERSION:
        raise _error("$.atlas_version", f"expected {ATLAS_VERSION}")
    name = _text(manifest["name"], "$.name")
    description = _text(manifest["description"], "$.description")

    records_directory = source / "records"
    records: list[dict[str, Any]] = []
    if records_directory.exists():
        if not records_directory.is_dir():
            raise CorpusError(f"{records_directory}: expected a directory")
        for entry in sorted(records_directory.iterdir(), key=lambda item: item.name):
            if entry.is_symlink() or not entry.is_dir():
                raise CorpusError(
                    f"{entry}: atlas records directory may contain only record directories"
                )
            records.append(load_record_bundle(entry))

    corpus = {
        "corpus_version": CORPUS_VERSION,
        "description": description,
        "name": name,
        "records": records,
    }
    validate_corpus(corpus)
    coverage_path = source / "coverage.json"
    if coverage_path.exists() or coverage_path.is_symlink():
        load_atlas_coverage(source, {record["id"] for record in records})
    return corpus


def _load_bound_bundle_file(
    bundle: Path,
    value: object,
    path: str,
    expected_name: str,
) -> Path:
    reference = _object(value, path)
    _exact_keys(reference, _FILE_REFERENCE_FIELDS, path)
    declared = _text(reference["path"], f"{path}.path")
    if declared != expected_name:
        raise _error(f"{path}.path", f"expected {expected_name!r}")
    expected_digest = _digest(reference["sha256"], f"{path}.sha256")
    source = bundle / declared
    if source.is_symlink() or not source.is_file():
        raise CorpusError(f"{source}: expected a regular file")
    if _file_sha256(source) != expected_digest:
        raise _error(f"{path}.sha256", f"does not match {declared}")
    return source


def _validate_symbolic_artifact(
    value: object,
    *,
    record_id: str,
    problem_id: str,
    raw_output_sha256: str,
    template: object,
) -> dict[str, object]:
    artifact = _object(value, "$.artifact")
    _exact_keys(artifact, _SYMBOLIC_ARTIFACT_FIELDS, "$.artifact")
    version = artifact["artifact_version"]
    if isinstance(version, bool) or version != SYMBOLIC_ARTIFACT_VERSION:
        raise _error(
            "$.artifact.artifact_version",
            f"expected {SYMBOLIC_ARTIFACT_VERSION}",
        )
    if artifact["artifact_kind"] != "symbolic_expression":
        raise _error("$.artifact.artifact_kind", "expected 'symbolic_expression'")
    if _text(artifact["artifact_id"], "$.artifact.artifact_id") != record_id:
        raise _error("$.artifact.artifact_id", "must match the record id")
    if _text(artifact["problem_id"], "$.artifact.problem_id") != problem_id:
        raise _error("$.artifact.problem_id", "must match the record problem_id")
    if _digest(artifact["raw_output_sha256"], "$.artifact.raw_output_sha256") != (
        raw_output_sha256
    ):
        raise _error("$.artifact.raw_output_sha256", "does not match raw-output.txt")

    fields = _object(artifact["fields"], "$.artifact.fields")
    expected_fields = set(template.field_names)
    missing = sorted(expected_fields - set(fields))
    unknown = sorted(set(fields) - expected_fields)
    if missing:
        raise _error("$.artifact.fields", f"missing field(s): {', '.join(missing)}")
    if unknown:
        raise _error("$.artifact.fields", f"unknown field(s): {', '.join(unknown)}")
    for name, expression in fields.items():
        _text(expression, f"$.artifact.fields.{name}")
    try:
        bind_symbolic_candidate(template, fields)
    except (TemplateError, TypeError, ValueError) as error:
        raise _error("$.artifact.fields", str(error)) from error
    return dict(artifact)


def _repository_relative_path(value: object, path: str) -> str:
    source = _text(value, path)
    relative = Path(source)
    if relative.is_absolute():
        raise _error(path, "expected a repository-relative path")
    if ".." in relative.parts:
        raise _error(path, "path escapes the repository root")
    return source


def _validate_transported_frozen_integrity_claim(
    value: object,
    *,
    artifact: object,
    artifact_sha256: str,
) -> dict[str, object]:
    """Check an integrity envelope without claiming its source files are present."""

    integrity = _object(value, "$.integrity")
    _exact_keys(integrity, _FROZEN_INTEGRITY_FIELDS, "$.integrity")
    version = integrity["schema_version"]
    if isinstance(version, bool) or version != 2:
        raise _error("$.integrity.schema_version", "Atlas v2 requires integrity version 2")
    _repository_relative_path(integrity["artifact_path"], "$.integrity.artifact_path")
    _text(integrity["base_revision"], "$.integrity.base_revision")
    training_run = _object(integrity["training_run"], "$.integrity.training_run")
    _exact_keys(training_run, {"executor", "run_id"}, "$.integrity.training_run")
    _text(training_run["executor"], "$.integrity.training_run.executor")
    _text(training_run["run_id"], "$.integrity.training_run.run_id")
    for field in ("artifact_sha256", "configuration_sha256", "weights_sha256"):
        _digest(integrity[field], f"$.integrity.{field}")

    expected = {
        "artifact_sha256": artifact_sha256,
        "configuration_sha256": canonical_frozen_configuration_sha256(artifact),
        "weights_sha256": artifact.weights_sha256,
    }
    for field, digest in expected.items():
        if integrity[field] != digest:
            raise _error(f"$.integrity.{field}", "digest mismatch")

    sources = _object(integrity["source_files_sha256"], "$.integrity.source_files_sha256")
    if not sources:
        raise _error(
            "$.integrity.source_files_sha256",
            "expected at least one bound source file",
        )
    if len(sources) > 64:
        raise _error("$.integrity.source_files_sha256", "source count exceeds 64")
    for relative, digest in sources.items():
        _repository_relative_path(relative, "$.integrity.source_files_sha256")
        _digest(digest, f"$.integrity.source_files_sha256.{relative}")
    training_script = artifact.training["script"]
    if training_script not in sources:
        raise _error(
            "$.integrity.source_files_sha256",
            "training script is not bound",
        )
    if sources[training_script] != artifact.training["script_sha256"]:
        raise _error("$.integrity.source_files_sha256", "training script digest mismatch")
    return dict(integrity)


def load_cross_artifact_record_bundle(path: str | Path) -> dict[str, Any]:
    """Load one Atlas v2 symbolic or frozen-callable record bundle.

    Validation establishes byte identity, provenance structure, and compatibility
    between the candidate-free template and artifact representation. It does not
    evaluate PDE obligations or create proof evidence.
    """

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise CorpusError(f"{source}: expected a regular record directory")
    metadata_path = source / "record.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise CorpusError(f"{metadata_path}: expected a regular file")
    metadata = _object(_load_cross_artifact_json(metadata_path), "$")
    _exact_keys(metadata, _CROSS_RECORD_FIELDS, "$")
    version = metadata["record_version"]
    if isinstance(version, bool) or version != CROSS_ARTIFACT_RECORD_VERSION:
        raise _error("$.record_version", f"expected {CROSS_ARTIFACT_RECORD_VERSION}")
    record_id = _text(metadata["id"], "$.id")
    if not _RECORD_ID.fullmatch(record_id):
        raise _error("$.id", "expected a lowercase corpus identifier")
    if source.name != record_id:
        raise CorpusError(f"{source}: directory name must match record id {record_id!r}")
    problem_id = _text(metadata["problem_id"], "$.problem_id")
    if not _RECORD_ID.fullmatch(problem_id):
        raise _error("$.problem_id", "expected a lowercase problem identifier")
    artifact_type = _text(metadata["artifact_type"], "$.artifact_type")
    if artifact_type not in _CROSS_ARTIFACT_TYPES:
        raise _error(
            "$.artifact_type",
            "Atlas v2 currently supports callable_model and symbolic_expression",
        )
    _validate_origin(metadata["origin"], "$.origin", allowed_kinds=_CROSS_ORIGIN_KINDS)
    _validate_annotation(metadata["annotation"], "$.annotation")

    expected_files = {
        "artifact": "artifact.json",
        "problem": "template.json",
        **(
            {"raw_output": "raw-output.txt"}
            if artifact_type == "symbolic_expression"
            else {"integrity": "integrity.json"}
        ),
    }
    files = _object(metadata["files"], "$.files")
    _exact_keys(files, set(expected_files), "$.files")
    bound = {
        name: _load_bound_bundle_file(source, files[name], f"$.files.{name}", filename)
        for name, filename in expected_files.items()
    }
    expected_entries = {"record.json", *expected_files.values()}
    actual_entries = {entry.name for entry in source.iterdir()}
    missing_entries = sorted(expected_entries - actual_entries)
    unknown_entries = sorted(actual_entries - expected_entries)
    if missing_entries:
        raise CorpusError(f"{source}: missing bundle file(s): {', '.join(missing_entries)}")
    if unknown_entries:
        raise CorpusError(f"{source}: unexpected bundle file(s): {', '.join(unknown_entries)}")

    template_payload = _load_cross_artifact_json(bound["problem"])
    try:
        template = template_from_dict(template_payload)
    except (TemplateError, TypeError, ValueError) as error:
        raise _error("$.template", str(error)) from error

    raw_output: str | None = None
    integrity_payload: dict[str, object] | None = None
    if artifact_type == "symbolic_expression":
        try:
            raw_output = bound["raw_output"].read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorpusError(f"{bound['raw_output']}: raw output must be valid UTF-8") from error
        if not raw_output.strip():
            raise CorpusError(f"{bound['raw_output']}: raw output must not be empty")
        artifact_payload = _validate_symbolic_artifact(
            _load_cross_artifact_json(bound["artifact"]),
            record_id=record_id,
            problem_id=problem_id,
            raw_output_sha256=_file_sha256(bound["raw_output"]),
            template=template,
        )
    else:
        try:
            artifact = load_frozen_callable(bound["artifact"])
            integrity_payload = _validate_transported_frozen_integrity_claim(
                _load_cross_artifact_json(bound["integrity"]),
                artifact=artifact,
                artifact_sha256=_file_sha256(bound["artifact"]),
            )
        except FrozenCallableError as error:
            raise _error("$.artifact", str(error)) from error
        if artifact.problem_id != problem_id:
            raise _error("$.artifact.problem_id", "must match the record problem_id")
        if artifact.input_names != tuple(template.variables):
            raise _error(
                "$.artifact.architecture.input_names",
                "must match template variables in order",
            )
        if artifact.output_names != tuple(template.field_names):
            raise _error(
                "$.artifact.architecture.output_names",
                "must match template field_names in order",
            )
        artifact_payload = frozen_callable_to_dict(artifact)

    return {
        "annotation": dict(metadata["annotation"]),
        "artifact": artifact_payload,
        "artifact_type": artifact_type,
        "files": dict(files),
        "id": record_id,
        "integrity": integrity_payload,
        "origin": dict(metadata["origin"]),
        "problem_id": problem_id,
        "raw_output": raw_output,
        "record_version": CROSS_ARTIFACT_RECORD_VERSION,
        "template": template_to_dict(template),
    }


def load_cross_artifact_atlas(path: str | Path) -> dict[str, Any]:
    """Load an Atlas v2 directory containing typed solution artifacts."""

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise CorpusError(f"{source}: expected a regular Atlas directory")
    manifest_path = source / "atlas.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CorpusError(f"{manifest_path}: expected a regular file")
    manifest = _object(_load_cross_artifact_json(manifest_path), "$")
    _exact_keys(manifest, {"atlas_version", "description", "name"}, "$")
    version = manifest["atlas_version"]
    if isinstance(version, bool) or version != CROSS_ARTIFACT_ATLAS_VERSION:
        raise _error("$.atlas_version", f"expected {CROSS_ARTIFACT_ATLAS_VERSION}")
    name = _text(manifest["name"], "$.name")
    description = _text(manifest["description"], "$.description")

    records_directory = source / "records"
    if records_directory.is_symlink() or not records_directory.is_dir():
        raise CorpusError(f"{records_directory}: expected a regular directory")
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for entry in sorted(records_directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_dir():
            raise CorpusError(
                f"{entry}: atlas records directory may contain only record directories"
            )
        record = load_cross_artifact_record_bundle(entry)
        if record["id"] in identifiers:
            raise CorpusError(f"{entry}: duplicate record id: {record['id']}")
        identifiers.add(record["id"])
        records.append(record)

    allowed_root_entries = {"README.md", "atlas.json", "coverage.json", "records"}
    unknown_root_entries = sorted(
        entry.name for entry in source.iterdir() if entry.name not in allowed_root_entries
    )
    if unknown_root_entries:
        raise CorpusError(
            f"{source}: unexpected atlas entry or entries: {', '.join(unknown_root_entries)}"
        )
    readme_path = source / "README.md"
    if readme_path.exists() or readme_path.is_symlink():
        if readme_path.is_symlink() or not readme_path.is_file():
            raise CorpusError(f"{readme_path}: expected a regular file")

    coverage_path = source / "coverage.json"
    if coverage_path.exists() or coverage_path.is_symlink():
        if coverage_path.is_symlink() or not coverage_path.is_file():
            raise CorpusError(f"{coverage_path}: expected a regular file")
        coverage = _load_cross_artifact_json(coverage_path)
        validate_atlas_coverage(coverage, identifiers)
        for record in records:
            declared = coverage["records"][record["id"]]["artifact_type"]
            if declared != record["artifact_type"]:
                raise CorpusError(
                    f"{coverage_path}: artifact_type for {record['id']!r} does not match record"
                )
    return {
        "atlas_version": CROSS_ARTIFACT_ATLAS_VERSION,
        "description": description,
        "name": name,
        "records": records,
    }


def load_corpus_source(path: str | Path) -> dict[str, Any]:
    """Load either a monolithic corpus file or a modular atlas directory."""

    source = Path(path)
    if source.is_dir():
        manifest = _object(_load_json(source / "atlas.json"), "$")
        if manifest.get("atlas_version") == CROSS_ARTIFACT_ATLAS_VERSION:
            return load_cross_artifact_atlas(source)
        return load_atlas(source)
    return load_corpus(source)


def dump_corpus(value: object, path: str | Path) -> None:
    """Validate and write deterministic corpus JSON."""

    validate_corpus(value)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dump_atlas(
    value: object,
    path: str | Path,
    *,
    coverage: object | None = None,
) -> None:
    """Validate and atomically write a new modular Atlas directory."""

    validate_corpus(value)
    corpus = _object(value, "$")
    if coverage is not None:
        validate_atlas_coverage(
            coverage,
            {record["id"] for record in corpus["records"]},
        )
    destination = Path(path)
    if destination.exists():
        raise CorpusError(f"{destination}: refusing to overwrite an existing path")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}-",
    ) as temporary:
        staged = Path(temporary) / destination.name
        records_directory = staged / "records"
        records_directory.mkdir(parents=True)
        manifest = {
            "atlas_version": ATLAS_VERSION,
            "description": corpus["description"],
            "name": corpus["name"],
        }
        (staged / "atlas.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if coverage is not None:
            (staged / "coverage.json").write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n"
            )
        for record in sorted(corpus["records"], key=lambda item: item["id"]):
            bundle = records_directory / record["id"]
            bundle.mkdir()
            metadata = {
                "annotation": record["annotation"],
                "id": record["id"],
                "origin": record["origin"],
                "output_sha256": record["output_sha256"],
            }
            (bundle / "record.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            (bundle / "case.json").write_text(
                json.dumps(record["case"], indent=2, sort_keys=True) + "\n"
            )
            (bundle / "raw-output.txt").write_bytes(record["raw_output"].encode())

        load_atlas(staged)
        if destination.exists():
            raise CorpusError(f"{destination}: refusing to overwrite an existing path")
        staged.replace(destination)
