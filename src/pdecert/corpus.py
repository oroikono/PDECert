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

from .schema import SCHEMA_VERSION, case_from_dict


CORPUS_VERSION = 1
ATLAS_VERSION = 1
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
_BUNDLE_FILES = frozenset({"case.json", "raw-output.txt", "record.json"})


class CorpusError(ValueError):
    """Raised when a candidate corpus does not follow the corpus schema."""


def output_sha256(raw_output: str) -> str:
    """Return the content digest stored with one raw generator output."""

    return hashlib.sha256(raw_output.encode()).hexdigest()


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


def _validate_origin(value: object, path: str) -> None:
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
    if kind not in ORIGIN_KINDS:
        raise _error(f"{path}.kind", f"expected one of: {', '.join(sorted(ORIGIN_KINDS))}")
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


def load_corpus_source(path: str | Path) -> dict[str, Any]:
    """Load either a monolithic corpus file or a modular atlas directory."""

    source = Path(path)
    if source.is_dir():
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
