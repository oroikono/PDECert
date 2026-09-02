import builtins
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pdecert import (
    FROZEN_CALLABLE_INTEGRITY_VERSION,
    FROZEN_CALLABLE_MAX_BYTES,
    FROZEN_CALLABLE_MAX_SOURCE_FILES,
    FROZEN_CALLABLE_VERSION,
    FrozenCallableArtifact,
    FrozenCallableError,
    canonical_frozen_configuration_sha256,
    canonical_frozen_weights_sha256,
    frozen_callable_from_dict,
    frozen_callable_to_dict,
    load_frozen_callable,
    materialize_frozen_callable,
    validate_frozen_callable_integrity,
    write_frozen_callable,
)


FIXTURE = Path("benchmarks/matched/burgers-classical-01/pinn.json")
INTEGRITY = Path("benchmarks/matched/burgers-classical-01/integrity.json")


def minimal_artifact() -> dict[str, object]:
    state = {
        "0.bias": [0.0, 0.0],
        "0.weight": [[1.0, 0.0], [0.0, 1.0]],
        "2.bias": [0.0],
        "2.weight": [[1.0, -1.0]],
    }
    return {
        "schema_version": 1,
        "artifact_id": "test-pinn",
        "artifact_kind": "trained_callable",
        "problem_id": "test-problem",
        "architecture": {
            "type": "dense_mlp",
            "activation": "tanh",
            "dtype": "float64",
            "input_names": ["x", "t"],
            "hidden_widths": [2],
            "output_names": ["u"],
        },
        "training": {
            "method": "physics_informed_collocation",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "seed": 1,
            "steps": 1,
            "device": "cpu",
            "torch_version": "test",
            "collocation": {"points": 3},
            "loss_weights": {"pde": 1.0},
            "final_losses": {"pde": 0.5},
            "script": "train.py",
            "script_sha256": "a" * 64,
            "generated_at": "2026-08-27T00:00:00+00:00",
        },
        "state_dict": state,
        "weights_sha256": canonical_frozen_weights_sha256(state),
    }


class FrozenCallableContractTests(unittest.TestCase):
    def test_valid_artifact_round_trips_and_writes_without_replacement(self):
        payload = minimal_artifact()
        artifact = frozen_callable_from_dict(payload)
        self.assertIsInstance(artifact, FrozenCallableArtifact)
        self.assertEqual(artifact.kind, "frozen_callable")
        self.assertEqual(artifact.input_names, ("x", "t"))
        self.assertEqual(artifact.output_names, ("u",))
        self.assertEqual(artifact.field_names, ("u",))
        self.assertEqual(frozen_callable_to_dict(artifact), payload)
        self.assertEqual(artifact.schema_version, FROZEN_CALLABLE_VERSION)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "frozen.json"
            write_frozen_callable(path, artifact)
            self.assertEqual(load_frozen_callable(path), artifact)
            with self.assertRaises(FileExistsError):
                write_frozen_callable(path, artifact)

    def test_exact_keys_are_required(self):
        payload = minimal_artifact()
        payload["unexpected"] = True
        with self.assertRaisesRegex(FrozenCallableError, "unknown field"):
            frozen_callable_from_dict(payload)

    def test_ragged_and_wrong_shapes_are_rejected(self):
        ragged = minimal_artifact()
        ragged["state_dict"]["0.weight"] = [[1.0, 0.0], [1.0]]
        ragged["weights_sha256"] = canonical_frozen_weights_sha256(ragged["state_dict"])
        with self.assertRaisesRegex(FrozenCallableError, "ragged"):
            frozen_callable_from_dict(ragged)

        wrong = minimal_artifact()
        wrong["state_dict"]["2.weight"] = [[1.0]]
        wrong["weights_sha256"] = canonical_frozen_weights_sha256(wrong["state_dict"])
        with self.assertRaisesRegex(FrozenCallableError, "expected shape"):
            frozen_callable_from_dict(wrong)

    def test_nonfinite_metadata_and_weights_are_rejected(self):
        metadata = minimal_artifact()
        metadata["training"]["final_losses"]["pde"] = math.nan
        with self.assertRaisesRegex(FrozenCallableError, "finite"):
            frozen_callable_from_dict(metadata)

        weights = minimal_artifact()
        weights["state_dict"]["0.bias"][0] = math.inf
        with self.assertRaisesRegex(FrozenCallableError, "finite"):
            frozen_callable_from_dict(weights)

    def test_nested_and_oversized_json_are_rejected_as_contract_errors(self):
        payload = minimal_artifact()
        nested: dict[str, object] = {}
        cursor = nested
        for _ in range(40):
            child: dict[str, object] = {}
            cursor["next"] = child
            cursor = child
        payload["training"]["collocation"] = nested
        with self.assertRaisesRegex(FrozenCallableError, "nesting exceeds"):
            frozen_callable_from_dict(payload)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text("{}")
            oversized = mock.Mock(st_size=FROZEN_CALLABLE_MAX_BYTES + 1)
            with mock.patch.object(Path, "stat", return_value=oversized):
                with self.assertRaisesRegex(FrozenCallableError, "byte limit"):
                    load_frozen_callable(path)

            deeply_nested = Path(directory) / "deep.json"
            serialized = json.dumps(minimal_artifact())
            deep_metadata = '{"next":' * 1200 + "{}" + "}" * 1200
            serialized = serialized.replace(
                '"collocation": {"points": 3}',
                f'"collocation": {deep_metadata}',
            )
            deeply_nested.write_text(serialized)
            with self.assertRaisesRegex(FrozenCallableError, "nesting"):
                load_frozen_callable(deeply_nested)

    def test_unsupported_activation_and_multiple_outputs_are_rejected(self):
        activation = minimal_artifact()
        activation["architecture"]["activation"] = "relu"
        with self.assertRaisesRegex(FrozenCallableError, "activation: unsupported"):
            frozen_callable_from_dict(activation)

        outputs = minimal_artifact()
        outputs["architecture"]["output_names"] = ["u", "v"]
        with self.assertRaisesRegex(FrozenCallableError, "exactly 1"):
            frozen_callable_from_dict(outputs)

    def test_weight_digest_mismatch_is_rejected(self):
        payload = minimal_artifact()
        payload["weights_sha256"] = "0" * 64
        with self.assertRaisesRegex(FrozenCallableError, "state_dict digest mismatch"):
            frozen_callable_from_dict(payload)

    def test_current_burgers_artifact_and_integrity_record_are_accepted(self):
        artifact = load_frozen_callable(FIXTURE)
        integrity = validate_frozen_callable_integrity(FIXTURE, INTEGRITY)
        self.assertEqual(artifact.problem_id, "burgers-traveling-wave-classical-01")
        self.assertEqual(artifact.weights_sha256, integrity["weights_sha256"])
        self.assertEqual(
            canonical_frozen_configuration_sha256(artifact),
            "b21df353a25165307fc2ff907595a87aff36d3875aeed057080f1f867160b130",
        )

    def test_portable_integrity_record_does_not_require_one_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "train.py"
            fixture = root / "frozen.json"
            integrity_path = root / "integrity.json"
            script.write_text("# deterministic training fixture\n")
            payload = minimal_artifact()
            payload["training"]["script"] = "train.py"
            payload["training"]["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
            write_frozen_callable(fixture, payload)
            artifact = load_frozen_callable(fixture)
            integrity = {
                "schema_version": FROZEN_CALLABLE_INTEGRITY_VERSION,
                "artifact_path": "frozen.json",
                "artifact_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "base_revision": "test-revision",
                "configuration_sha256": canonical_frozen_configuration_sha256(artifact),
                "training_run": {"executor": "local-cpu", "run_id": "test-seed-1"},
                "source_files_sha256": {
                    "train.py": payload["training"]["script_sha256"],
                },
                "weights_sha256": artifact.weights_sha256,
            }
            integrity_path.write_text(json.dumps(integrity))
            observed = validate_frozen_callable_integrity(
                fixture,
                integrity_path,
                repository_root=root,
            )
            self.assertEqual(observed["training_run"]["executor"], "local-cpu")

    def test_integrity_rejects_path_traversal_and_digest_mismatch(self):
        integrity = json.loads(INTEGRITY.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrity.json"
            traversing = copy.deepcopy(integrity)
            traversing["artifact_path"] = "../pinn.json"
            path.write_text(json.dumps(traversing))
            with self.assertRaisesRegex(FrozenCallableError, "escapes"):
                validate_frozen_callable_integrity(FIXTURE, path)

            mismatched = copy.deepcopy(integrity)
            mismatched["artifact_sha256"] = "0" * 64
            path.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(FrozenCallableError, "artifact_sha256: digest mismatch"):
                validate_frozen_callable_integrity(FIXTURE, path)

            too_many_sources = copy.deepcopy(integrity)
            too_many_sources["source_files_sha256"] = {
                f"source-{index}.py": "0" * 64
                for index in range(FROZEN_CALLABLE_MAX_SOURCE_FILES + 1)
            }
            path.write_text(json.dumps(too_many_sources))
            with self.assertRaisesRegex(FrozenCallableError, "source count exceeds"):
                validate_frozen_callable_integrity(FIXTURE, path)

    def test_missing_torch_is_reported_only_at_materialization(self):
        artifact = frozen_callable_from_dict(minimal_artifact())
        original_import = builtins.__import__

        def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=reject_torch):
            with self.assertRaisesRegex(RuntimeError, "autodiff"):
                materialize_frozen_callable(artifact)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "materialization requires the optional PyTorch dependency")
class FrozenCallableMaterializationTests(unittest.TestCase):
    def test_materializer_builds_an_evaluation_only_callable_candidate(self):
        artifact = load_frozen_callable(FIXTURE)
        candidate = materialize_frozen_callable(artifact)
        self.assertEqual(candidate.field_names, ("u",))
        self.assertEqual(candidate.dtype, "float64")
        self.assertEqual(candidate.device, "cpu")
        model = candidate.fields[0][1]
        points = torch.zeros((2, 2), dtype=torch.float64)
        self.assertEqual(tuple(model(points).shape), (2, 1))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
