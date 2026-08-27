import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.burgers_pinn_fixture import (
    FixtureError,
    canonical_weights_sha256,
    load_manifest,
    validate_integrity_manifest,
    validate_manifest,
)


FIXTURE = Path("benchmarks/matched/burgers-classical-01/pinn.json")
INTEGRITY = Path("benchmarks/matched/burgers-classical-01/integrity.json")


def minimal_manifest():
    state = {
        "0.bias": [0.0, 0.0],
        "0.weight": [[0.0, 0.0], [0.0, 0.0]],
        "2.bias": [0.0],
        "2.weight": [[0.0, 0.0]],
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
            "collocation": {},
            "loss_weights": {},
            "final_losses": {},
            "script": "train.py",
            "script_sha256": "a" * 64,
            "generated_at": "2026-08-27T00:00:00+00:00",
        },
        "state_dict": state,
        "weights_sha256": canonical_weights_sha256(state),
    }


class BurgersFixtureContractTests(unittest.TestCase):
    def test_valid_restricted_manifest(self):
        self.assertEqual(validate_manifest(minimal_manifest())["artifact_id"], "test-pinn")

    def test_invalid_weight_shape_is_rejected(self):
        manifest = minimal_manifest()
        manifest["state_dict"]["0.weight"] = [[0.0, 0.0]]
        manifest["weights_sha256"] = canonical_weights_sha256(manifest["state_dict"])
        with self.assertRaisesRegex(FixtureError, "expected shape"):
            validate_manifest(manifest)

    def test_unsupported_activation_is_rejected(self):
        manifest = copy.deepcopy(minimal_manifest())
        manifest["architecture"]["activation"] = "relu"
        with self.assertRaisesRegex(FixtureError, "activation: unsupported"):
            validate_manifest(manifest)

    def test_committed_fixture_and_source_digests_match(self):
        integrity = validate_integrity_manifest(FIXTURE, INTEGRITY)
        self.assertEqual(integrity["weights_sha256"], load_manifest(FIXTURE)["weights_sha256"])

    def test_integrity_rejects_an_artifact_digest_mismatch(self):
        integrity = json.loads(INTEGRITY.read_text())
        integrity["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(FixtureError, "artifact_sha256: digest mismatch"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "integrity.json"
                path.write_text(json.dumps(integrity))
                validate_integrity_manifest(FIXTURE, path)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "trained fixture requires PyTorch")
class TrainedBurgersPairTests(unittest.TestCase):
    def test_frozen_artifact_has_valid_provenance_and_expected_lane_evidence(self):
        from experiments.trained_burgers_pair import build_case
        from pdecert import EvidenceLevel, LaneVerificationOptions, Status, verify_matched_case

        self.assertTrue(FIXTURE.is_file(), "the committed trained fixture is missing")
        manifest = load_manifest(FIXTURE)
        self.assertEqual(manifest["training"]["method"], "physics_informed_collocation")
        case, _ = build_case(FIXTURE)
        report = verify_matched_case(
            case,
            options={
                "symbolic-exact": LaneVerificationOptions(symbolic_timeout=2.0),
                "trained-pinn": LaneVerificationOptions(tolerance=1e-3, samples_per_axis=7),
            },
        )
        self.assertEqual(report.reports["symbolic-exact"].status, Status.PROVED)
        self.assertEqual(report.reports["trained-pinn"].status, Status.REFUTED)
        self.assertEqual(
            report.reports["trained-pinn"].decision_evidence,
            EvidenceLevel.EMPIRICAL,
        )
        self.assertNotIn("status", report.to_dict())

    def test_committed_result_names_the_same_frozen_weights(self):
        result = json.loads(Path("results/trained-burgers-pair.json").read_text())
        manifest = load_manifest(FIXTURE)
        self.assertEqual(result["fixture"]["weights_sha256"], manifest["weights_sha256"])
        self.assertEqual(
            result["fixture"]["source_files_sha256"],
            json.loads(INTEGRITY.read_text())["source_files_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
