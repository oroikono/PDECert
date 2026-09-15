"""Portable, non-executing frozen callable artifacts.

Version 1 deliberately supports one narrow architecture: a CPU ``float64``
dense MLP with ``tanh`` hidden activations.  Loading and validation are JSON-
only operations; PyTorch is imported only when an artifact is materialized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from .artifacts import CallableCandidate


FROZEN_CALLABLE_VERSION = 1
FROZEN_CALLABLE_INTEGRITY_VERSION = 2
FROZEN_CALLABLE_MAX_BYTES = 4_000_000
FROZEN_CALLABLE_MAX_JSON_DEPTH = 32
FROZEN_CALLABLE_MAX_JSON_VALUES = 200_000
FROZEN_CALLABLE_MAX_SOURCE_FILES = 64

_TOP_LEVEL_FIELDS = {
    "architecture",
    "artifact_id",
    "artifact_kind",
    "problem_id",
    "schema_version",
    "state_dict",
    "training",
    "weights_sha256",
}
_ARCHITECTURE_FIELDS = {
    "activation",
    "dtype",
    "hidden_widths",
    "input_names",
    "output_names",
    "type",
}
_TRAINING_FIELDS = {
    "collocation",
    "device",
    "final_losses",
    "generated_at",
    "learning_rate",
    "loss_weights",
    "method",
    "optimizer",
    "script",
    "script_sha256",
    "seed",
    "steps",
    "torch_version",
}
_LEGACY_INTEGRITY_FIELDS = {
    "artifact_path",
    "artifact_sha256",
    "base_revision",
    "configuration_sha256",
    "euler_job_id",
    "schema_version",
    "source_files_sha256",
    "weights_sha256",
}
_INTEGRITY_FIELDS = (_LEGACY_INTEGRITY_FIELDS - {"euler_job_id"}) | {"training_run"}
_TRAINING_RUN_FIELDS = {"executor", "run_id"}
_CONFIGURATION_TRAINING_FIELDS = (
    "collocation",
    "device",
    "learning_rate",
    "loss_weights",
    "method",
    "optimizer",
    "seed",
    "steps",
    "torch_version",
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

__all__ = [
    "FROZEN_CALLABLE_INTEGRITY_VERSION",
    "FROZEN_CALLABLE_MAX_BYTES",
    "FROZEN_CALLABLE_MAX_JSON_DEPTH",
    "FROZEN_CALLABLE_MAX_JSON_VALUES",
    "FROZEN_CALLABLE_MAX_SOURCE_FILES",
    "FROZEN_CALLABLE_VERSION",
    "FrozenCallableArtifact",
    "FrozenCallableError",
    "canonical_frozen_configuration_sha256",
    "canonical_frozen_weights_sha256",
    "frozen_callable_from_dict",
    "frozen_callable_to_dict",
    "load_frozen_callable",
    "materialize_frozen_callable",
    "validate_frozen_callable_integrity",
    "write_frozen_callable",
]


class FrozenCallableError(ValueError):
    """Raised when a frozen callable artifact is malformed or unsupported."""


@dataclass(frozen=True)
class FrozenCallableArtifact:
    """Validated, immutable JSON representation of a trained callable."""

    kind: ClassVar[str] = "frozen_callable"

    schema_version: int
    artifact_id: str
    artifact_kind: str
    problem_id: str
    architecture: Mapping[str, object]
    training: Mapping[str, object]
    state_dict: Mapping[str, object]
    weights_sha256: str

    def __post_init__(self) -> None:
        payload = _validate_manifest(_artifact_payload(self))
        object.__setattr__(self, "schema_version", FROZEN_CALLABLE_VERSION)
        object.__setattr__(self, "artifact_id", payload["artifact_id"])
        object.__setattr__(self, "artifact_kind", payload["artifact_kind"])
        object.__setattr__(self, "problem_id", payload["problem_id"])
        object.__setattr__(self, "architecture", _freeze_json(payload["architecture"]))
        object.__setattr__(self, "training", _freeze_json(payload["training"]))
        object.__setattr__(self, "state_dict", _freeze_json(payload["state_dict"]))
        object.__setattr__(self, "weights_sha256", payload["weights_sha256"])

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the model input names in stable column order."""

        return tuple(self.architecture["input_names"])

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the single model output name."""

        return tuple(self.architecture["output_names"])

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return candidate field names for the public artifact protocol."""

        return self.output_names


def _exact_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise FrozenCallableError(f"{path}: missing field(s): {', '.join(missing)}")
    if unknown:
        raise FrozenCallableError(f"{path}: unknown field(s): {', '.join(unknown)}")


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrozenCallableError(f"{path}: expected an object")
    if any(not isinstance(key, str) for key in value):
        raise FrozenCallableError(f"{path}: object keys must be strings")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenCallableError(f"{path}: expected a non-empty string")
    return value


def _digest(value: object, path: str) -> str:
    digest = _text(value, path)
    if _SHA256.fullmatch(digest) is None:
        raise FrozenCallableError(f"{path}: expected a lowercase SHA-256 digest")
    return digest


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FrozenCallableError(f"{path}: expected a positive integer")
    return value


def _finite_number(value: object, path: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenCallableError(f"{path}: expected a finite number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0):
        qualifier = "finite positive" if positive else "finite"
        raise FrozenCallableError(f"{path}: expected a {qualifier} number")
    return value


def _validate_json_value(
    value: object,
    path: str,
    active: set[int] | None = None,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> None:
    """Reject non-JSON and non-finite metadata before canonicalization."""

    if depth > FROZEN_CALLABLE_MAX_JSON_DEPTH:
        raise FrozenCallableError(
            f"{path}: JSON nesting exceeds {FROZEN_CALLABLE_MAX_JSON_DEPTH} levels"
        )
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > FROZEN_CALLABLE_MAX_JSON_VALUES:
        raise FrozenCallableError(
            f"{path}: JSON value count exceeds {FROZEN_CALLABLE_MAX_JSON_VALUES}"
        )

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FrozenCallableError(f"{path}: JSON numbers must be finite")
        return
    if not isinstance(value, (dict, list)):
        raise FrozenCallableError(f"{path}: expected a JSON value")

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise FrozenCallableError(f"{path}: cyclic values are not valid JSON")
    active.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise FrozenCallableError(f"{path}: object keys must be strings")
                _validate_json_value(
                    item,
                    f"{path}.{key}",
                    active,
                    depth=depth + 1,
                    budget=budget,
                )
        else:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    f"{path}[{index}]",
                    active,
                    depth=depth + 1,
                    budget=budget,
                )
    finally:
        active.remove(identity)


def _tensor_shape(value: object, path: str) -> tuple[int, ...]:
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        raise FrozenCallableError(f"{path}: expected finite numeric tensor values")
    if isinstance(value, (int, float)):
        _finite_number(value, path)
        return ()
    if not value:
        raise FrozenCallableError(f"{path}: tensor dimensions must not be empty")
    shapes = [_tensor_shape(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(shapes)) != 1:
        raise FrozenCallableError(f"{path}: ragged tensor values are not supported")
    return (len(value), *shapes[0])


def _expected_state_shapes(
    input_count: int,
    hidden_widths: list[int],
) -> dict[str, tuple[int, ...]]:
    widths = [input_count, *hidden_widths, 1]
    shapes: dict[str, tuple[int, ...]] = {}
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        module_index = index * 2
        shapes[f"{module_index}.weight"] = (output_width, input_width)
        shapes[f"{module_index}.bias"] = (output_width,)
    return shapes


def _validate_names(
    value: object,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        if minimum == maximum:
            expectation = f"exactly {minimum} identifier"
        else:
            expectation = f"between {minimum} and {maximum} identifiers"
        raise FrozenCallableError(f"{path}: expected {expectation}")
    if any(not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None for name in value):
        raise FrozenCallableError(f"{path}: names must be identifiers")
    if len(value) != len(set(value)):
        raise FrozenCallableError(f"{path}: names must be unique")
    return value


def _validate_manifest(value: object) -> dict[str, object]:
    manifest = _object(value, "$")
    _validate_json_value(manifest, "$")
    _exact_keys(manifest, _TOP_LEVEL_FIELDS, "$")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise FrozenCallableError(f"$.schema_version: expected {FROZEN_CALLABLE_VERSION}")
    for field in ("artifact_id", "artifact_kind", "problem_id"):
        _text(manifest[field], f"$.{field}")
    if manifest["artifact_kind"] != "trained_callable":
        raise FrozenCallableError("$.artifact_kind: expected 'trained_callable'")

    architecture = _object(manifest["architecture"], "$.architecture")
    _exact_keys(architecture, _ARCHITECTURE_FIELDS, "$.architecture")
    supported = {
        "activation": "tanh",
        "dtype": "float64",
        "type": "dense_mlp",
    }
    for field, expected in supported.items():
        if architecture[field] != expected:
            raise FrozenCallableError(f"$.architecture.{field}: unsupported value")
    input_names = _validate_names(
        architecture["input_names"],
        "$.architecture.input_names",
        minimum=1,
        maximum=8,
    )
    _validate_names(
        architecture["output_names"],
        "$.architecture.output_names",
        minimum=1,
        maximum=1,
    )
    hidden_widths = architecture["hidden_widths"]
    if (
        not isinstance(hidden_widths, list)
        or not 1 <= len(hidden_widths) <= 4
        or any(isinstance(width, bool) or not isinstance(width, int) for width in hidden_widths)
        or any(not 1 <= width <= 128 for width in hidden_widths)
    ):
        raise FrozenCallableError(
            "$.architecture.hidden_widths: expected one to four widths between 1 and 128"
        )

    training = _object(manifest["training"], "$.training")
    _exact_keys(training, _TRAINING_FIELDS, "$.training")
    for field in ("device", "generated_at", "method", "optimizer", "script", "torch_version"):
        _text(training[field], f"$.training.{field}")
    if training["device"] != "cpu":
        raise FrozenCallableError("$.training.device: only CPU artifacts are supported")
    _positive_integer(training["seed"], "$.training.seed")
    _positive_integer(training["steps"], "$.training.steps")
    _finite_number(training["learning_rate"], "$.training.learning_rate", positive=True)
    for field in ("collocation", "final_losses", "loss_weights"):
        _object(training[field], f"$.training.{field}")
    _digest(training["script_sha256"], "$.training.script_sha256")

    state_dict = _object(manifest["state_dict"], "$.state_dict")
    expected_shapes = _expected_state_shapes(len(input_names), hidden_widths)
    _exact_keys(state_dict, set(expected_shapes), "$.state_dict")
    for name, expected_shape in expected_shapes.items():
        observed = _tensor_shape(state_dict[name], f"$.state_dict.{name}")
        if observed != expected_shape:
            raise FrozenCallableError(
                f"$.state_dict.{name}: expected shape {expected_shape}, observed {observed}"
            )

    digest = _digest(manifest["weights_sha256"], "$.weights_sha256")
    if canonical_frozen_weights_sha256(state_dict) != digest:
        raise FrozenCallableError("$.weights_sha256: state_dict digest mismatch")
    return manifest


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _artifact_payload(artifact: FrozenCallableArtifact) -> dict[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "artifact_id": artifact.artifact_id,
        "artifact_kind": artifact.artifact_kind,
        "problem_id": artifact.problem_id,
        "architecture": _thaw_json(artifact.architecture),
        "training": _thaw_json(artifact.training),
        "state_dict": _thaw_json(artifact.state_dict),
        "weights_sha256": artifact.weights_sha256,
    }


def canonical_frozen_weights_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash canonical JSON tensor values; the digest establishes identity only."""

    if not isinstance(state_dict, Mapping):
        raise FrozenCallableError("state_dict: expected an object")
    payload = _thaw_json(state_dict)
    state = _object(payload, "state_dict")
    _validate_json_value(state, "state_dict")
    try:
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FrozenCallableError("state_dict: values must be finite JSON tensors") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_frozen_configuration_sha256(
    manifest_or_artifact: Mapping[str, object] | FrozenCallableArtifact,
) -> str:
    """Hash declared training inputs; the digest establishes identity only."""

    if isinstance(manifest_or_artifact, FrozenCallableArtifact):
        manifest = _artifact_payload(manifest_or_artifact)
    elif isinstance(manifest_or_artifact, Mapping):
        manifest = dict(manifest_or_artifact)
    else:
        raise FrozenCallableError("manifest_or_artifact: expected an artifact or object")
    validated = _validate_manifest(manifest)
    training = _object(validated["training"], "$.training")
    payload = {
        "architecture": validated["architecture"],
        "artifact_id": validated["artifact_id"],
        "problem_id": validated["problem_id"],
        "training": {name: training[name] for name in _CONFIGURATION_TRAINING_FIELDS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def frozen_callable_from_dict(value: object) -> FrozenCallableArtifact:
    """Validate a JSON object without importing or executing model code."""

    manifest = _validate_manifest(value)
    return FrozenCallableArtifact(
        schema_version=FROZEN_CALLABLE_VERSION,
        artifact_id=manifest["artifact_id"],
        artifact_kind=manifest["artifact_kind"],
        problem_id=manifest["problem_id"],
        architecture=_freeze_json(manifest["architecture"]),
        training=_freeze_json(manifest["training"]),
        state_dict=_freeze_json(manifest["state_dict"]),
        weights_sha256=manifest["weights_sha256"],
    )


def frozen_callable_to_dict(artifact: FrozenCallableArtifact) -> dict[str, object]:
    """Return a fresh JSON-compatible dictionary for a validated artifact."""

    if not isinstance(artifact, FrozenCallableArtifact):
        raise TypeError("artifact must be a FrozenCallableArtifact")
    payload = _artifact_payload(artifact)
    _validate_manifest(payload)
    return payload


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise FrozenCallableError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _nonfinite_json_constant(value: str) -> object:
    raise FrozenCallableError(f"non-finite JSON number is not supported: {value}")


def _load_json(path: Path) -> object:
    try:
        size = path.stat().st_size
        if size > FROZEN_CALLABLE_MAX_BYTES:
            raise FrozenCallableError(
                f"{path}: file exceeds the {FROZEN_CALLABLE_MAX_BYTES}-byte limit"
            )
        text = path.read_text(encoding="utf-8")
    except FrozenCallableError:
        raise
    except (OSError, UnicodeError) as error:
        raise FrozenCallableError(f"{path}: could not read file: {error}") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_nonfinite_json_constant,
        )
    except json.JSONDecodeError as error:
        raise FrozenCallableError(f"{path}: invalid JSON: {error.msg}") from error
    except RecursionError as error:
        raise FrozenCallableError(f"{path}: JSON nesting exceeds the decoder limit") from error


def load_frozen_callable(path: str | Path) -> FrozenCallableArtifact:
    """Load and validate a frozen callable JSON artifact without importing PyTorch."""

    return frozen_callable_from_dict(_load_json(Path(path)))


def write_frozen_callable(
    path: str | Path,
    payload: Mapping[str, object] | FrozenCallableArtifact,
) -> None:
    """Validate and atomically create an artifact without replacing existing evidence."""

    if isinstance(payload, FrozenCallableArtifact):
        artifact = payload
    else:
        artifact = frozen_callable_from_dict(payload)
    rendered = (
        json.dumps(
            frozen_callable_to_dict(artifact),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_frozen_callable(artifact: FrozenCallableArtifact) -> CallableCandidate:
    """Build the restricted evaluation-only PyTorch model for an artifact."""

    if not isinstance(artifact, FrozenCallableArtifact):
        raise TypeError("artifact must be a FrozenCallableArtifact")
    payload = frozen_callable_to_dict(artifact)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PDECert with the 'autodiff' extra") from error

    architecture = payload["architecture"]
    widths = [
        len(architecture["input_names"]),
        *architecture["hidden_widths"],
        1,
    ]
    modules = []
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        modules.append(torch.nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            modules.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*modules).to(dtype=torch.float64, device="cpu")
    tensors = {
        name: torch.tensor(value, dtype=torch.float64, device="cpu")
        for name, value in payload["state_dict"].items()
    }
    model.load_state_dict(tensors, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return CallableCandidate.from_mapping(
        {artifact.output_names[0]: model},
        dtype="float64",
        device="cpu",
    )


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FrozenCallableError(f"{path}: could not read file: {error}") from error
    return digest.hexdigest()


def _safe_repository_file(root: Path, value: object, path: str) -> Path:
    relative = Path(_text(value, path))
    if relative.is_absolute():
        raise FrozenCallableError(f"{path}: expected a repository-relative path")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise FrozenCallableError(f"{path}: path escapes the repository root")
    if not resolved.is_file():
        raise FrozenCallableError(f"{path}: file does not exist")
    return resolved


def validate_frozen_callable_integrity(
    fixture_path: str | Path,
    integrity_path: str | Path,
    repository_root: str | Path = ".",
) -> dict[str, object]:
    """Validate artifact identity and its digest-bound training source record."""

    fixture = Path(fixture_path)
    integrity_source = Path(integrity_path)
    integrity = _object(_load_json(integrity_source), "$")
    _validate_json_value(integrity, "$")
    version = integrity.get("schema_version")
    if type(version) is not int or version not in {1, FROZEN_CALLABLE_INTEGRITY_VERSION}:
        raise FrozenCallableError(
            f"$.schema_version: expected legacy version 1 or {FROZEN_CALLABLE_INTEGRITY_VERSION}"
        )
    if version == 1:
        _exact_keys(integrity, _LEGACY_INTEGRITY_FIELDS, "$")
        _text(integrity["euler_job_id"], "$.euler_job_id")
    else:
        _exact_keys(integrity, _INTEGRITY_FIELDS, "$")
        training_run = _object(integrity["training_run"], "$.training_run")
        _exact_keys(training_run, _TRAINING_RUN_FIELDS, "$.training_run")
        _text(training_run["executor"], "$.training_run.executor")
        _text(training_run["run_id"], "$.training_run.run_id")
    for field in ("artifact_path", "base_revision"):
        _text(integrity[field], f"$.{field}")
    for field in (
        "artifact_sha256",
        "configuration_sha256",
        "weights_sha256",
    ):
        _digest(integrity[field], f"$.{field}")

    root = Path(repository_root).resolve()
    declared_fixture = _safe_repository_file(root, integrity["artifact_path"], "$.artifact_path")
    if declared_fixture != fixture.resolve():
        raise FrozenCallableError("$.artifact_path: does not identify the supplied fixture")
    artifact = load_frozen_callable(fixture)
    observed = {
        "artifact_sha256": _file_sha256(fixture),
        "configuration_sha256": canonical_frozen_configuration_sha256(artifact),
        "weights_sha256": artifact.weights_sha256,
    }
    for field, value in observed.items():
        if integrity[field] != value:
            raise FrozenCallableError(f"$.{field}: digest mismatch")

    sources = _object(integrity["source_files_sha256"], "$.source_files_sha256")
    if not sources:
        raise FrozenCallableError("$.source_files_sha256: expected at least one bound source file")
    if len(sources) > FROZEN_CALLABLE_MAX_SOURCE_FILES:
        raise FrozenCallableError(
            f"$.source_files_sha256: source count exceeds {FROZEN_CALLABLE_MAX_SOURCE_FILES}"
        )
    for relative, expected in sources.items():
        source = _safe_repository_file(root, relative, "$.source_files_sha256")
        digest = _digest(expected, f"$.source_files_sha256.{relative}")
        if _file_sha256(source) != digest:
            raise FrozenCallableError(f"$.source_files_sha256.{relative}: digest mismatch")

    training_script = _text(artifact.training["script"], "$.training.script")
    if training_script not in sources:
        raise FrozenCallableError("$.source_files_sha256: training script is not bound")
    if sources[training_script] != artifact.training["script_sha256"]:
        raise FrozenCallableError("$.training.script_sha256: digest mismatch")
    return integrity
