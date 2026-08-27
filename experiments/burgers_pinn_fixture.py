"""Restricted JSON fixture support for the trained Burgers PINN benchmark lane."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEMA_VERSION = 1
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
_INTEGRITY_FIELDS = {
    "artifact_path",
    "artifact_sha256",
    "base_revision",
    "configuration_sha256",
    "euler_job_id",
    "schema_version",
    "source_files_sha256",
    "weights_sha256",
}


class FixtureError(ValueError):
    """Raised when a frozen callable fixture is malformed or unsupported."""


def _exact_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise FixtureError(f"{path}: missing field(s): {', '.join(missing)}")
    if unknown:
        raise FixtureError(f"{path}: unknown field(s): {', '.join(unknown)}")


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FixtureError(f"{path}: expected an object")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{path}: expected a non-empty string")
    return value


def canonical_weights_sha256(state_dict: dict[str, object]) -> str:
    """Hash the exact JSON-compatible tensor values independently of formatting."""

    encoded = json.dumps(state_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def canonical_configuration_sha256(manifest: Mapping[str, object]) -> str:
    """Hash only the declared inputs that determine the training run."""

    training = manifest["training"]
    payload = {
        "architecture": manifest["architecture"],
        "artifact_id": manifest["artifact_id"],
        "problem_id": manifest["problem_id"],
        "training": {
            name: training[name]
            for name in (
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
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one file without normalizing or reserializing its bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_state_shapes(hidden_widths: list[int]) -> dict[str, tuple[int, ...]]:
    """Return the only state layout supported by the dense-tanh loader."""

    widths = [2, *hidden_widths, 1]
    shapes: dict[str, tuple[int, ...]] = {}
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        module_index = index * 2
        shapes[f"{module_index}.weight"] = (output_width, input_width)
        shapes[f"{module_index}.bias"] = (output_width,)
    return shapes


def _tensor_shape(value: object, path: str) -> tuple[int, ...]:
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        raise FixtureError(f"{path}: expected finite numeric tensor values")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise FixtureError(f"{path}: expected finite numeric tensor values")
        return ()
    if not value:
        raise FixtureError(f"{path}: tensor dimensions must not be empty")
    shapes = [_tensor_shape(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(shapes)) != 1:
        raise FixtureError(f"{path}: ragged tensor values are not supported")
    return (len(value), *shapes[0])


def validate_manifest(value: object) -> dict[str, object]:
    """Validate one non-executing, architecture-restricted trained artifact."""

    manifest = _object(value, "$")
    _exact_keys(manifest, _TOP_LEVEL_FIELDS, "$")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise FixtureError(f"$.schema_version: expected {SCHEMA_VERSION}")
    for field in ("artifact_id", "artifact_kind", "problem_id"):
        _text(manifest[field], f"$.{field}")
    if manifest["artifact_kind"] != "trained_callable":
        raise FixtureError("$.artifact_kind: expected 'trained_callable'")

    architecture = _object(manifest["architecture"], "$.architecture")
    _exact_keys(architecture, _ARCHITECTURE_FIELDS, "$.architecture")
    supported = {
        "activation": "tanh",
        "dtype": "float64",
        "input_names": ["x", "t"],
        "output_names": ["u"],
        "type": "dense_mlp",
    }
    for field, expected in supported.items():
        if architecture[field] != expected:
            raise FixtureError(f"$.architecture.{field}: unsupported value")
    hidden_widths = architecture["hidden_widths"]
    if (
        not isinstance(hidden_widths, list)
        or not 1 <= len(hidden_widths) <= 4
        or any(isinstance(width, bool) or not isinstance(width, int) for width in hidden_widths)
        or any(not 1 <= width <= 128 for width in hidden_widths)
    ):
        raise FixtureError(
            "$.architecture.hidden_widths: expected one to four widths between 1 and 128"
        )

    training = _object(manifest["training"], "$.training")
    _exact_keys(training, _TRAINING_FIELDS, "$.training")
    for field in ("device", "generated_at", "method", "optimizer", "script", "torch_version"):
        _text(training[field], f"$.training.{field}")
    if training["method"] != "physics_informed_collocation":
        raise FixtureError("$.training.method: unsupported value")
    if training["optimizer"] != "Adam":
        raise FixtureError("$.training.optimizer: unsupported value")
    if training["device"] != "cpu":
        raise FixtureError("$.training.device: only deterministic CPU fixtures are supported")
    for field in ("seed", "steps"):
        item = training[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise FixtureError(f"$.training.{field}: expected a positive integer")
    learning_rate = training["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise FixtureError("$.training.learning_rate: expected a finite positive number")
    for field in ("collocation", "final_losses", "loss_weights"):
        _object(training[field], f"$.training.{field}")
    script_sha256 = _text(training["script_sha256"], "$.training.script_sha256")
    if len(script_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in script_sha256
    ):
        raise FixtureError("$.training.script_sha256: expected a lowercase SHA-256 digest")

    state_dict = _object(manifest["state_dict"], "$.state_dict")
    expected_shapes = expected_state_shapes(hidden_widths)
    _exact_keys(state_dict, set(expected_shapes), "$.state_dict")
    for name, expected_shape in expected_shapes.items():
        observed = _tensor_shape(state_dict[name], f"$.state_dict.{name}")
        if observed != expected_shape:
            raise FixtureError(
                f"$.state_dict.{name}: expected shape {expected_shape}, observed {observed}"
            )

    digest = _text(manifest["weights_sha256"], "$.weights_sha256")
    if canonical_weights_sha256(state_dict) != digest:
        raise FixtureError("$.weights_sha256: state_dict digest mismatch")
    return manifest


def load_manifest(path: str | Path) -> dict[str, object]:
    """Load and validate one fixture manifest."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise FixtureError(f"{source}: invalid JSON: {error.msg}") from error
    return validate_manifest(payload)


def validate_integrity_manifest(
    fixture_path: str | Path,
    integrity_path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> dict[str, object]:
    """Verify fixture, configuration, weights, and bound source files."""

    fixture = Path(fixture_path)
    integrity_source = Path(integrity_path)
    try:
        integrity = _object(json.loads(integrity_source.read_text()), "$")
    except json.JSONDecodeError as error:
        raise FixtureError(f"{integrity_source}: invalid JSON: {error.msg}") from error
    _exact_keys(integrity, _INTEGRITY_FIELDS, "$")
    if integrity["schema_version"] != SCHEMA_VERSION:
        raise FixtureError(f"$.schema_version: expected {SCHEMA_VERSION}")
    for field in (
        "artifact_path",
        "artifact_sha256",
        "base_revision",
        "configuration_sha256",
        "euler_job_id",
        "weights_sha256",
    ):
        _text(integrity[field], f"$.{field}")

    root = Path(repository_root).resolve()
    declared_fixture = _safe_source_path(root, integrity["artifact_path"], "$.artifact_path")
    if declared_fixture != fixture.resolve():
        raise FixtureError("$.artifact_path: does not identify the supplied fixture")
    manifest = load_manifest(fixture)
    observed = {
        "artifact_sha256": file_sha256(fixture),
        "configuration_sha256": canonical_configuration_sha256(manifest),
        "weights_sha256": manifest["weights_sha256"],
    }
    for field, value in observed.items():
        if integrity[field] != value:
            raise FixtureError(f"$.{field}: digest mismatch")

    sources = _object(integrity["source_files_sha256"], "$.source_files_sha256")
    if not sources:
        raise FixtureError("$.source_files_sha256: expected at least one bound source file")
    for relative, expected in sources.items():
        source = _safe_source_path(root, relative, "$.source_files_sha256")
        digest = _text(expected, f"$.source_files_sha256.{relative}")
        if file_sha256(source) != digest:
            raise FixtureError(f"$.source_files_sha256.{relative}: digest mismatch")

    training = manifest["training"]
    training_script = _text(training["script"], "$.training.script")
    if training_script not in sources:
        raise FixtureError("$.source_files_sha256: training script is not bound")
    if sources[training_script] != training["script_sha256"]:
        raise FixtureError("$.training.script_sha256: digest mismatch")
    return integrity


def _safe_source_path(root: Path, value: object, path: str) -> Path:
    relative = Path(_text(value, path))
    if relative.is_absolute():
        raise FixtureError(f"{path}: expected a repository-relative path")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise FixtureError(f"{path}: path escapes the repository root")
    if not resolved.is_file():
        raise FixtureError(f"{path}: file does not exist")
    return resolved


def build_dense_tanh_model(torch, hidden_widths: list[int]):
    """Construct the only executable architecture accepted by this fixture lane."""

    widths = [2, *hidden_widths, 1]
    modules = []
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        modules.append(torch.nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            modules.append(torch.nn.Tanh())
    return torch.nn.Sequential(*modules)


def load_frozen_model(path: str | Path):
    """Materialize a validated JSON fixture as an evaluation-only PyTorch module."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PDECert with the 'autodiff' extra") from error

    manifest = load_manifest(path)
    hidden_widths = manifest["architecture"]["hidden_widths"]
    model = build_dense_tanh_model(torch, hidden_widths).to(dtype=torch.float64, device="cpu")
    tensors = {
        name: torch.tensor(value, dtype=torch.float64, device="cpu")
        for name, value in manifest["state_dict"].items()
    }
    model.load_state_dict(tensors, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, manifest


def write_new_manifest(path: str | Path, payload: object) -> None:
    """Validate and atomically create a fixture without replacing evidence."""

    destination = Path(path)
    validated = validate_manifest(payload)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing fixture: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a frozen fixture and its companion integrity record."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("integrity", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    arguments = parser.parse_args(argv)
    try:
        validate_integrity_manifest(
            arguments.fixture,
            arguments.integrity,
            repository_root=arguments.repository_root,
        )
    except (FixtureError, OSError) as error:
        print(f"burgers_pinn_fixture: {error}", file=sys.stderr)
        return 2
    print(f"Validated {arguments.fixture} and {arguments.integrity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
