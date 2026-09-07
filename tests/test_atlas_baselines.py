import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import sympy as sp
from jsonschema import Draft202012Validator

from pdecert.atlas_baselines import (
    ATLAS_BASELINE_REPORT_VERSION,
    AtlasBaselineError,
    BaselineOutcome,
    BaselineResult,
    BaselineWitness,
    FixedCollocationBaseline,
    evaluate_atlas_baseline,
)
from pdecert.corpus import load_cross_artifact_atlas
from pdecert.corpus import cross_artifact_atlas_sha256
from pdecert.cli import INPUT_ERROR, main
from pdecert.templates import bind_symbolic_candidate, template_from_dict


ATLAS = Path("corpus/matched")
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SCHEMA = Path("schema/atlas-baseline-report-v1.schema.json")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA.read_text())
    return Draft202012Validator(schema)


class _ExampleAdapter:
    adapter_id = "example_empirical"
    adapter_version = 1
    accepted_artifact_types = ("symbolic_expression",)
    accepted_solution_semantics = ("classical_strong",)

    def __init__(self):
        self.calls = []

    def configuration(self):
        return {"tolerance": 1e-9}

    def evaluate_record(self, record):
        self.calls.append(record["id"])
        return BaselineResult(
            BaselineOutcome.PASS,
            "EMPIRICAL_PASS",
            "EMPIRICAL",
            evaluations=1,
            max_absolute_residual=0.0,
        )


class BaselineExtensionTests(unittest.TestCase):
    def test_external_adapter_receives_only_its_declared_scope(self):
        adapter = _ExampleAdapter()
        report = evaluate_atlas_baseline(ATLAS, adapter)

        self.assertEqual(adapter.calls, [SYMBOLIC_ID])
        self.assertEqual([row["outcome"] for row in report["records"]], ["pass", "unsupported"])
        self.assertEqual(list(_validator().iter_errors(report)), [])
        self.assertEqual(
            report["atlas"]["sha256"], cross_artifact_atlas_sha256(load_cross_artifact_atlas(ATLAS))
        )

        adapter.accepted_solution_semantics = ("external_semantics",)
        adapter.calls.clear()
        report = evaluate_atlas_baseline(ATLAS, adapter)
        self.assertEqual(adapter.calls, [])
        self.assertTrue(all(row["outcome"] == "unsupported" for row in report["records"]))

    def test_adapter_cannot_rewrite_the_evaluated_artifact_or_report_identity(self):
        class MutatingAdapter(_ExampleAdapter):
            def evaluate_record(self, record):
                record["artifact"]["fields"]["u"] = "0"
                record["id"] = "rewritten-record"
                return super().evaluate_record(record)

        atlas = load_cross_artifact_atlas(ATLAS)
        original = copy.deepcopy(atlas)
        with patch("pdecert.atlas_baselines.load_cross_artifact_atlas", return_value=atlas):
            with self.assertRaisesRegex(AtlasBaselineError, "modified.*record"):
                evaluate_atlas_baseline(ATLAS, MutatingAdapter(), record_ids=[SYMBOLIC_ID])
        self.assertEqual(atlas, original)

    def test_pass_constructor_rejects_nonfinite_residual_and_nonnull_reason(self):
        for overrides in (
            {"max_absolute_residual": "infinity"},
            {"reason": ""},
            {"reason": False},
            {"reason": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                BaselineResult(
                    **{
                        "outcome": BaselineOutcome.PASS,
                        "evidence_kind": "EMPIRICAL_PASS",
                        "evidence_level": "EMPIRICAL",
                        "evaluations": 1,
                        "max_absolute_residual": 0.0,
                        **overrides,
                    }
                )

    def test_failure_constructor_requires_a_matching_typed_witness(self):
        witness = BaselineWitness("PDE", "D(u, x)", {"x": 0.5}, 2.0)
        for overrides in (
            {"witness": {}},
            {"witness": "counterexample"},
            {"max_absolute_residual": 1.0},
            {"reason": ""},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                BaselineResult(
                    **{
                        "outcome": BaselineOutcome.FAIL,
                        "evidence_kind": "NUMERICAL_THRESHOLD_EXCEEDANCE",
                        "evidence_level": "EMPIRICAL",
                        "evaluations": 1,
                        "max_absolute_residual": 2.0,
                        "witness": witness,
                        **overrides,
                    }
                )

    def test_witness_owns_immutable_sample_inputs(self):
        inputs = {"x": 0.5}
        witness = BaselineWitness("PDE", "D(u, x)", inputs, 2.0)
        inputs["x"] = float("nan")
        self.assertEqual(witness.to_dict()["sampled_inputs"], {"x": 0.5})
        with self.assertRaises(TypeError):
            witness.sampled_inputs["x"] = 1.0

    def test_bad_adapter_scope_metadata_fails_with_a_structured_error(self):
        for values in ((["symbolic_expression"],), ("symbolic_expression", "symbolic_expression")):
            adapter = _ExampleAdapter()
            adapter.accepted_artifact_types = values
            with self.subTest(values=values), self.assertRaises(AtlasBaselineError):
                evaluate_atlas_baseline(ATLAS, adapter)

    def test_reserved_adapter_id_requires_its_documented_configuration(self):
        class MisidentifiedAdapter(_ExampleAdapter):
            adapter_id = "fixed_collocation"

        with self.assertRaisesRegex(AtlasBaselineError, "fixed_collocation"):
            evaluate_atlas_baseline(ATLAS, MisidentifiedAdapter())

        class FutureVersion(FixedCollocationBaseline):
            adapter_version = 2

        with self.assertRaisesRegex(AtlasBaselineError, "contract"):
            evaluate_atlas_baseline(ATLAS, FutureVersion())

        for key, value in (("include_conditions", 1), ("max_evaluations", 2_000_000)):

            class ChangedConfiguration(FixedCollocationBaseline):
                def configuration(self):
                    return {**super().configuration(), key: value}

            with self.subTest(key=key), self.assertRaisesRegex(AtlasBaselineError, "contract"):
                evaluate_atlas_baseline(ATLAS, ChangedConfiguration())

    def test_external_failure_result_matches_the_schema(self):
        class ThresholdAdapter(_ExampleAdapter):
            def evaluate_record(self, record):
                return BaselineResult(
                    BaselineOutcome.FAIL,
                    "NUMERICAL_THRESHOLD_EXCEEDANCE",
                    "EMPIRICAL",
                    evaluations=1,
                    max_absolute_residual=2.0,
                    witness=BaselineWitness("PDE", "D(u, x)", {"x": 0.5}, 2.0),
                )

        report = evaluate_atlas_baseline(ATLAS, ThresholdAdapter())
        self.assertEqual([row["outcome"] for row in report["records"]], ["fail", "unsupported"])
        self.assertEqual(list(_validator().iter_errors(report)), [])


class AtlasBaselineTests(unittest.TestCase):
    def test_fixed_collocation_pass_is_empirical_and_not_a_proof(self):
        report = evaluate_atlas_baseline(
            ATLAS,
            FixedCollocationBaseline(points_per_axis=5, tolerance=1e-9),
            record_ids=[SYMBOLIC_ID],
        )

        self.assertEqual(report["baseline_report_version"], ATLAS_BASELINE_REPORT_VERSION)
        self.assertEqual(
            report["evidence_policy"],
            "method_specific_empirical_diagnostics_no_proof",
        )
        self.assertNotIn("status", report)
        self.assertNotIn("accuracy", report)
        record = report["records"][0]
        self.assertEqual(record["outcome"], BaselineOutcome.PASS.value)
        self.assertEqual(record["evidence_kind"], "EMPIRICAL_PASS")
        self.assertEqual(record["evidence_level"], "EMPIRICAL")
        self.assertGreater(record["evaluations"], 0)
        self.assertIsNone(record["witness"])

    def test_fixed_collocation_failure_has_a_replayable_condition_witness(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        record = copy.deepcopy(
            next(record for record in atlas["records"] if record["id"] == SYMBOLIC_ID)
        )
        record["artifact"]["fields"]["u"] = "0"

        result = FixedCollocationBaseline(points_per_axis=3).evaluate_record(record)

        self.assertEqual(result.outcome, BaselineOutcome.FAIL)
        self.assertEqual(result.evidence_kind, "NUMERICAL_THRESHOLD_EXCEEDANCE")
        self.assertIsNotNone(result.witness)
        witness = result.witness
        assert witness is not None
        self.assertIn(
            witness.constraint,
            {"initial condition", "left boundary", "right boundary"},
        )
        self.assertIn("At(u", witness.constraint_source)
        self.assertEqual(len(witness.sampled_inputs), 1)
        fixed_variable = "t" if ", t," in witness.constraint_source else "x"
        self.assertNotIn(fixed_variable, witness.sampled_inputs)
        self.assertGreater(float(witness.absolute_residual), 0)

    def test_fully_fixed_boundary_witness_retains_the_original_surface(self):
        template = json.loads(Path("examples/heat-template.json").read_text())
        template["pde_residuals"] = [{"name": "zero PDE", "expression": "0*u"}]
        template["conditions"] = [{"name": "right boundary", "expression": "At(u, x, 1) - 1"}]
        record = {
            "artifact": {"fields": {"u": "0"}},
            "artifact_type": "symbolic_expression",
            "id": "right-boundary-failure",
            "template": template,
        }

        result = FixedCollocationBaseline(points_per_axis=3).evaluate_record(record)

        self.assertEqual(result.outcome, BaselineOutcome.FAIL)
        witness = result.witness
        assert witness is not None
        self.assertEqual(witness.constraint_source, "At(u, x, 1) - 1")
        self.assertEqual(witness.sampled_inputs, {})

    def test_callable_record_abstains_as_unsupported(self):
        report = evaluate_atlas_baseline(
            ATLAS,
            FixedCollocationBaseline(),
            record_ids=[CALLABLE_ID],
        )

        record = report["records"][0]
        self.assertEqual(record["outcome"], BaselineOutcome.UNSUPPORTED.value)
        self.assertEqual(record["evidence_kind"], "ABSTENTION")
        self.assertIsNone(record["evidence_level"])
        self.assertIsNone(record["evaluations"])
        self.assertIsNone(record["max_absolute_residual"])
        self.assertIn("symbolic_expression", record["reason"])

    def test_grid_alias_can_pass_even_when_the_pde_fails_between_points(self):
        template = json.loads(Path("examples/heat-template.json").read_text())
        fields = {"u": "exp(-pi**2*t)*sin(pi*x) + sin(4*pi*x)"}
        record = {
            "artifact": {"fields": fields},
            "artifact_type": "symbolic_expression",
            "id": "fixed-grid-alias",
            "template": template,
        }

        result = FixedCollocationBaseline(points_per_axis=5).evaluate_record(record)

        self.assertEqual(result.outcome, BaselineOutcome.PASS)
        case = bind_symbolic_candidate(template_from_dict(template), fields)
        x, t = case.problem.variables
        off_grid = case.problem.pde_residuals[0].residual.subs({x: sp.Rational(1, 8), t: 0})
        self.assertNotEqual(sp.simplify(off_grid), 0)

    def test_roundoff_exceedance_is_not_labeled_a_counterexample(self):
        template = json.loads(Path("examples/heat-template.json").read_text())
        template["pde_residuals"][0]["expression"] = "sin(x)**2 + cos(x)**2 - 1"
        record = {
            "artifact": {"fields": {"u": "exp(-pi**2*t)*sin(pi*x)"}},
            "artifact_type": "symbolic_expression",
            "id": "roundoff-identity",
            "template": template,
        }

        result = FixedCollocationBaseline(
            decimal_precision=30,
            points_per_axis=5,
            tolerance=1e-40,
        ).evaluate_record(record)

        self.assertEqual(result.outcome, BaselineOutcome.FAIL)
        self.assertEqual(result.evidence_kind, "NUMERICAL_THRESHOLD_EXCEEDANCE")

    def test_wide_finite_domain_uses_overflow_safe_grid_interpolation(self):
        template = {
            "conditions": [],
            "domains": {"x": [-1e308, 1e308]},
            "field_names": ["u"],
            "name": "wide-domain derivative",
            "parameters": {},
            "pde_residuals": [{"name": "u_x", "expression": "D(u, x)"}],
            "solution_semantics": "classical_strong",
            "template_version": 1,
            "variables": ["x"],
        }
        record = {
            "artifact": {"fields": {"u": "x**2"}},
            "artifact_type": "symbolic_expression",
            "id": "wide-finite-domain",
            "template": template,
        }

        result = FixedCollocationBaseline(points_per_axis=5).evaluate_record(record)

        self.assertEqual(result.outcome, BaselineOutcome.FAIL)
        witness = result.witness
        assert witness is not None
        self.assertTrue(all(math.isfinite(value) for value in witness.sampled_inputs.values()))
        self.assertTrue(all(-1e308 <= value <= 1e308 for value in witness.sampled_inputs.values()))

    def test_mixed_report_matches_the_public_schema(self):
        report = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())

        self.assertEqual(list(_validator().iter_errors(report)), [])
        self.assertEqual(
            [record["outcome"] for record in report["records"]],
            ["pass", "unsupported"],
        )

    def test_public_schema_rejects_proof_language_and_missing_witness(self):
        report = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())
        report["records"][0]["outcome"] = "PROVED"
        self.assertTrue(list(_validator().iter_errors(report)))

        report = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())
        record = report["records"][0]
        record["outcome"] = "fail"
        record["evidence_kind"] = "NUMERICAL_THRESHOLD_EXCEEDANCE"
        self.assertTrue(list(_validator().iter_errors(report)))

    def test_configuration_and_record_selection_are_strict(self):
        with self.assertRaisesRegex(ValueError, "points_per_axis"):
            FixedCollocationBaseline(points_per_axis=1)
        with self.assertRaisesRegex(ValueError, "decimal_precision"):
            FixedCollocationBaseline(decimal_precision=101)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            FixedCollocationBaseline(tolerance=float("nan"))
        atlas = load_cross_artifact_atlas(ATLAS)
        symbolic = next(record for record in atlas["records"] if record["id"] == SYMBOLIC_ID)
        with self.assertRaisesRegex(AtlasBaselineError, "1000000-evaluation limit"):
            FixedCollocationBaseline(points_per_axis=1_001).evaluate_record(symbolic)

        huge_integer = json.loads(Path("examples/heat-template.json").read_text())
        huge_integer["variables"].append("n")
        huge_integer["domains"]["n"] = [1.0, 1e308]
        huge_integer["parameters"]["n"] = ["integer", "positive"]
        huge_integer["conditions"][1]["expression"] = "At(u, x, 0) + n"
        huge_record = {
            "artifact": {"fields": {"u": "exp(-pi**2*t)*sin(pi*x)"}},
            "artifact_type": "symbolic_expression",
            "id": "huge-integer-domain",
            "template": huge_integer,
        }
        huge_result = FixedCollocationBaseline(points_per_axis=5).evaluate_record(huge_record)
        self.assertEqual(huge_result.outcome, BaselineOutcome.FAIL)
        with self.assertRaisesRegex(AtlasBaselineError, "unknown Atlas record"):
            evaluate_atlas_baseline(
                ATLAS,
                FixedCollocationBaseline(),
                record_ids=["missing-record"],
            )
        with self.assertRaisesRegex(AtlasBaselineError, "duplicate record"):
            evaluate_atlas_baseline(
                ATLAS,
                FixedCollocationBaseline(),
                record_ids=[SYMBOLIC_ID, SYMBOLIC_ID],
            )

    def test_version_one_atlas_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(AtlasBaselineError, "expected 2"):
            evaluate_atlas_baseline("corpus/community", FixedCollocationBaseline())

    def test_cli_writes_a_reproducible_baseline_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "baseline.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "corpus",
                        "baseline",
                        str(ATLAS),
                        "--record",
                        SYMBOLIC_ID,
                        "--points-per-axis",
                        "3",
                        "--decimal-precision",
                        "40",
                        "--output",
                        str(output_path),
                    ]
                )
            payload = json.loads(output_path.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(payload["adapter"]["id"], "fixed_collocation")
        self.assertEqual(payload["adapter"]["configuration"]["points_per_axis"], 3)
        self.assertEqual(payload["adapter"]["configuration"]["decimal_precision"], 40)
        self.assertEqual(payload["records"][0]["outcome"], "pass")

    def test_cli_reports_invalid_atlas_input(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(["corpus", "baseline", "corpus/community"])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("expected 2", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
