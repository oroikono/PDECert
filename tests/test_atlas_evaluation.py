import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from pdecert import (
    ATLAS_EVALUATION_VERSION,
    AtlasEvaluationError,
    AtlasEvaluationOptions,
    Status,
    atlas_evaluation_sha256,
    canonical_frozen_weights_sha256,
    cross_artifact_atlas_sha256,
    evaluate_cross_artifact_atlas,
    load_atlas_evaluation,
    load_cross_artifact_atlas,
    review_source_sha256,
    summarize_atlas_evaluation,
    validate_atlas_evaluation,
)
from pdecert.atlas_evaluation import _evaluate_record


ATLAS = Path("corpus/matched")
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _evaluation_validator() -> Draft202012Validator:
    evaluation_schema = json.loads(Path("schema/atlas-evaluation-v1.schema.json").read_text())
    report_schema = json.loads(Path("schema/report-v1.schema.json").read_text())
    registry = Registry().with_resource(
        report_schema["$id"],
        Resource.from_contents(report_schema),
    )
    return Draft202012Validator(evaluation_schema, registry=registry)


def _summary_validator() -> Draft202012Validator:
    schema = json.loads(Path("schema/atlas-evaluation-summary-v1.schema.json").read_text())
    return Draft202012Validator(schema)


def _mixed_evaluation_without_torch() -> dict[str, object]:
    evaluation = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
    callable_record = copy.deepcopy(evaluation["records"][0])
    callable_record.update(
        {
            "artifact_id": "test-callable",
            "artifact_type": "callable_model",
            "evaluator": "pdecert_autodiff",
            "record_id": "test-callable-01",
        }
    )
    callable_record["report"] = {
        "aggregation_policy_version": 1,
        "decision_evidence": None,
        "evidence_events": [
            {
                "bound": None,
                "checker": "autodiff_residual",
                "detail": "No sampled violation exceeded tolerance.",
                "kind": "EMPIRICAL_PASS",
                "level": "EMPIRICAL",
                "obligation_id": "pde_residuals[0]",
                "outcome": "OBSERVED_PASS",
                "witness": None,
            }
        ],
        "exact_checks": {},
        "incomplete_reasons": {"pde_residuals[0]": "finite sampled success is not proof"},
        "max_sampled_residual": 0.0,
        "report_version": 1,
        "status": "INCONCLUSIVE",
        "witness": None,
    }
    evaluation["records"].append(callable_record)
    evaluation["runtime"]["torch_version"] = "test"
    return evaluation


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

    def test_empty_atlas_is_rejected_before_emitting_an_invalid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "atlas.json").write_text(
                json.dumps(
                    {
                        "atlas_version": 2,
                        "description": "empty test Atlas",
                        "name": "empty test Atlas",
                    }
                )
            )
            with self.assertRaisesRegex(AtlasEvaluationError, "no records to evaluate"):
                evaluate_cross_artifact_atlas(root)

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

    def test_public_schema_accepts_a_genuine_symbolic_evaluation(self):
        payload = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])

        self.assertEqual(list(_evaluation_validator().iter_errors(payload)), [])

    def test_public_schema_rejects_evidence_transferred_to_a_callable_record(self):
        symbolic = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        tampered = copy.deepcopy(symbolic)
        tampered["records"][0]["artifact_type"] = "callable_model"
        tampered["records"][0]["evaluator"] = "pdecert_autodiff"

        errors = list(_evaluation_validator().iter_errors(tampered))

        self.assertTrue(errors)
        self.assertTrue(any(error.path and error.path[-1] == "status" for error in errors))

    def test_public_schema_rejects_an_evaluator_artifact_mismatch(self):
        payload = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        payload["records"][0]["evaluator"] = "pdecert_autodiff"

        errors = list(_evaluation_validator().iter_errors(payload))

        self.assertTrue(errors)
        self.assertTrue(any(error.path and error.path[-1] == "evaluator" for error in errors))

    def test_evaluation_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"evaluation_version": 1, "evaluation_version": 1}')

            with self.assertRaisesRegex(AtlasEvaluationError, "duplicate object key"):
                load_atlas_evaluation(path)

    def test_evaluation_validator_rejects_callable_proof_evidence(self):
        evaluation = _mixed_evaluation_without_torch()
        evaluation["records"][1]["report"] = copy.deepcopy(evaluation["records"][0]["report"])

        with self.assertRaisesRegex(AtlasEvaluationError, "cannot be PROVED"):
            validate_atlas_evaluation(evaluation)

    def test_evaluation_validator_requires_callable_runtime_provenance(self):
        evaluation = _mixed_evaluation_without_torch()
        del evaluation["runtime"]["torch_version"]

        with self.assertRaisesRegex(AtlasEvaluationError, "torch_version"):
            validate_atlas_evaluation(evaluation)
        self.assertTrue(list(_evaluation_validator().iter_errors(evaluation)))

    def test_evaluation_validator_rejects_exact_checks_on_callable_rows(self):
        evaluation = _mixed_evaluation_without_torch()
        evaluation["records"][1]["report"]["exact_checks"] = {"PDE": "zero"}

        with self.assertRaisesRegex(AtlasEvaluationError, "symbolic exact checks"):
            validate_atlas_evaluation(evaluation)
        self.assertTrue(list(_evaluation_validator().iter_errors(evaluation)))

    def test_evaluation_validator_wraps_extreme_numeric_values(self):
        evaluation = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        evaluation["records"][0]["report"]["max_sampled_residual"] = 10**4_000

        with self.assertRaisesRegex(AtlasEvaluationError, "report"):
            validate_atlas_evaluation(evaluation)

        evaluation = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        evaluation["options"]["symbolic_tolerance"] = 10**4_000
        with self.assertRaisesRegex(AtlasEvaluationError, "finite and positive"):
            validate_atlas_evaluation(evaluation)

    def test_evaluation_digest_normalizes_equivalent_numeric_options(self):
        integer_options = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        integer_options["options"]["symbolic_tolerance"] = 1
        float_options = copy.deepcopy(integer_options)
        float_options["options"]["symbolic_tolerance"] = 1.0

        self.assertEqual(
            atlas_evaluation_sha256(integer_options),
            atlas_evaluation_sha256(float_options),
        )

    def test_summary_is_descriptive_digest_bound_and_per_record(self):
        evaluation = _mixed_evaluation_without_torch()

        summary = summarize_atlas_evaluation(evaluation)

        self.assertEqual(summary["summary_version"], 1)
        self.assertEqual(
            summary["evidence_policy"],
            "descriptive_per_record_no_truth_labels",
        )
        self.assertNotIn("status", summary)
        self.assertNotIn("accuracy", summary)
        self.assertEqual(
            summary["source"]["evaluation_sha256"], atlas_evaluation_sha256(evaluation)
        )
        self.assertEqual(summary["coverage"]["records"], 2)
        self.assertEqual(summary["coverage"]["problems"], 1)
        self.assertEqual(summary["coverage"]["cross_artifact_problems"], 1)
        self.assertEqual(
            summary["coverage"]["artifact_types"],
            {"callable_model": 1, "symbolic_expression": 1},
        )
        self.assertEqual(summary["outcomes"]["statuses"], {"INCONCLUSIVE": 1, "PROVED": 1})
        self.assertEqual(summary["outcomes"]["decision_evidence"], {"EXACT": 1, "NONE": 1})
        self.assertEqual(
            [record["record_id"] for record in summary["problems"][0]["records"]],
            [SYMBOLIC_ID, "test-callable-01"],
        )
        self.assertEqual(list(_summary_validator().iter_errors(summary)), [])

    def test_summary_schema_rejects_callable_proof_evidence(self):
        summary = summarize_atlas_evaluation(_mixed_evaluation_without_torch())
        callable_record = summary["problems"][0]["records"][1]
        callable_record["status"] = "PROVED"
        callable_record["decision_evidence"] = "EXACT"

        self.assertTrue(list(_summary_validator().iter_errors(summary)))


if __name__ == "__main__":
    unittest.main()
