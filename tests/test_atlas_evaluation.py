import importlib.util
import json
import unittest
from pathlib import Path

from pdecert import (
    ATLAS_EVALUATION_VERSION,
    AtlasEvaluationError,
    AtlasEvaluationOptions,
    Status,
    canonical_frozen_weights_sha256,
    cross_artifact_atlas_sha256,
    evaluate_cross_artifact_atlas,
    load_cross_artifact_atlas,
    review_source_sha256,
)
from pdecert.atlas_evaluation import _evaluate_record


ATLAS = Path("corpus/matched")
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _zero_callable_record() -> dict[str, object]:
    state = {
        "0.bias": [0.0],
        "0.weight": [[0.0, 0.0]],
        "2.bias": [0.0],
        "2.weight": [[0.0]],
    }
    artifact = {
        "schema_version": 1,
        "artifact_id": "zero-heat-pinn",
        "artifact_kind": "trained_callable",
        "problem_id": "homogeneous-heat-01",
        "architecture": {
            "type": "dense_mlp",
            "activation": "tanh",
            "dtype": "float64",
            "input_names": ["x", "t"],
            "hidden_widths": [1],
            "output_names": ["u"],
        },
        "training": {
            "method": "physics_informed_collocation",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "seed": 7,
            "steps": 1,
            "device": "cpu",
            "torch_version": "test",
            "collocation": {"points": 4},
            "loss_weights": {"pde": 1.0},
            "final_losses": {"pde": 0.0},
            "script": "train.py",
            "script_sha256": "a" * 64,
            "generated_at": "2026-09-02T00:00:00+00:00",
        },
        "state_dict": state,
        "weights_sha256": canonical_frozen_weights_sha256(state),
    }
    template = {
        "template_version": 1,
        "name": "homogeneous heat equation",
        "solution_semantics": "classical_strong",
        "variables": ["x", "t"],
        "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
        "parameters": {},
        "field_names": ["u"],
        "pde_residuals": [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}],
        "conditions": [
            {"name": "initial condition", "expression": "At(u, t, 0)"},
            {"name": "left boundary", "expression": "At(u, x, 0)"},
            {"name": "right boundary", "expression": "At(u, x, 1)"},
        ],
    }
    return {
        "artifact": artifact,
        "artifact_type": "callable_model",
        "id": "zero-heat-pinn-01",
        "problem_id": "homogeneous-heat-01",
        "template": template,
    }


class AtlasEvaluationTests(unittest.TestCase):
    def test_symbolic_record_produces_exact_per_record_evidence(self):
        payload = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])

        self.assertEqual(payload["evaluation_version"], ATLAS_EVALUATION_VERSION)
        self.assertEqual(payload["evidence_policy"], "per_record_no_aggregation")
        self.assertNotIn("status", payload)
        self.assertNotIn("decision_evidence", payload)
        self.assertEqual(len(payload["records"]), 1)
        record = payload["records"][0]
        self.assertEqual(record["record_id"], SYMBOLIC_ID)
        self.assertEqual(record["artifact_type"], "symbolic_expression")
        self.assertEqual(record["report"]["status"], Status.PROVED.value)
        self.assertEqual(record["report"]["decision_evidence"], "EXACT")

    def test_evaluation_and_review_bind_the_same_loaded_atlas_content(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        payload = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])

        self.assertEqual(payload["atlas"]["sha256"], cross_artifact_atlas_sha256(atlas))
        self.assertEqual(payload["atlas"]["sha256"], review_source_sha256(atlas))

    def test_record_selection_rejects_unknown_and_duplicate_ids(self):
        with self.assertRaisesRegex(AtlasEvaluationError, "unknown Atlas record"):
            evaluate_cross_artifact_atlas(ATLAS, record_ids=["missing-record"])
        with self.assertRaisesRegex(AtlasEvaluationError, "duplicate record"):
            evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID, SYMBOLIC_ID])
        with self.assertRaisesRegex(AtlasEvaluationError, "sequence of record identifiers"):
            evaluate_cross_artifact_atlas(ATLAS, record_ids=SYMBOLIC_ID)

    def test_version_one_atlas_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(AtlasEvaluationError, "expected 2"):
            evaluate_cross_artifact_atlas("corpus/community")

    def test_options_reject_non_reproducible_values(self):
        with self.assertRaisesRegex(ValueError, "samples_per_axis"):
            AtlasEvaluationOptions(samples_per_axis=1)
        with self.assertRaisesRegex(ValueError, "callable_tolerance"):
            AtlasEvaluationOptions(callable_tolerance=float("nan"))
        with self.assertRaisesRegex(ValueError, "symbolic_tolerance"):
            AtlasEvaluationOptions(symbolic_tolerance=True)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is an optional dependency")
    def test_frozen_callable_counterexample_remains_empirical(self):
        payload = evaluate_cross_artifact_atlas(
            ATLAS,
            record_ids=[CALLABLE_ID],
            options=AtlasEvaluationOptions(callable_tolerance=1e-3, samples_per_axis=6),
        )

        record = payload["records"][0]
        self.assertEqual(record["report"]["status"], Status.REFUTED.value)
        self.assertEqual(record["report"]["decision_evidence"], "EMPIRICAL")
        self.assertIsNotNone(record["report"]["witness"])

    @unittest.skipUnless(HAS_TORCH, "PyTorch is an optional dependency")
    def test_passing_frozen_callable_remains_inconclusive(self):
        record, torch_version = _evaluate_record(
            _zero_callable_record(),
            AtlasEvaluationOptions(callable_tolerance=1e-9, samples_per_axis=3),
        )

        self.assertIsNotNone(torch_version)
        self.assertEqual(record["report"]["status"], Status.INCONCLUSIVE.value)
        self.assertIsNone(record["report"]["decision_evidence"])
        self.assertTrue(record["report"]["incomplete_reasons"])

    def test_public_schema_is_well_formed_json(self):
        self.assertIsInstance(
            json.loads(Path("schema/atlas-evaluation-v1.schema.json").read_text()),
            dict,
        )


if __name__ == "__main__":
    unittest.main()
