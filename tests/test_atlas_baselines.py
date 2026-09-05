import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import sympy as sp
from jsonschema import Draft202012Validator

from pdecert.atlas_baselines import (
    ATLAS_BASELINE_REPORT_VERSION,
    AtlasBaselineError,
    BaselineOutcome,
    FixedCollocationBaseline,
    evaluate_atlas_baseline,
)
from pdecert.corpus import load_cross_artifact_atlas
from pdecert.cli import INPUT_ERROR, main
from pdecert.templates import bind_symbolic_candidate, template_from_dict


ATLAS = Path("corpus/matched")
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SCHEMA = Path("schema/atlas-baseline-report-v1.schema.json")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA.read_text())
    return Draft202012Validator(schema)


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
