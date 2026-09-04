import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from experiments.adversarial_heat import build_cases
from pdecert import VerificationCase, dump_case, evaluate_cross_artifact_atlas
from pdecert.cli import INPUT_ERROR, main


class CommandLineTests(unittest.TestCase):
    def _write_case(self, directory: str, case_index: int) -> Path:
        experiment = build_cases()[case_index]
        path = Path(directory) / f"{experiment.name}.json"
        dump_case(VerificationCase(experiment.problem, (experiment.candidate,)), path)
        return path

    def test_proved_case_prints_json_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_case(directory, 0)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["verify", str(path)])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["report"]["status"], "PROVED")
        self.assertEqual(payload["report"]["report_version"], 1)
        self.assertEqual(payload["report"]["aggregation_policy_version"], 1)
        self.assertEqual(payload["report"]["decision_evidence"], "EXACT")
        self.assertTrue(payload["report"]["evidence_events"])
        self.assertEqual(payload["report"]["incomplete_reasons"], {})
        self.assertEqual(payload["problem"], "exact_heat_solution")
        self.assertEqual(payload["schema_version"], 3)

    def test_refuted_case_returns_one_and_includes_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_case(directory, 2)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["verify", str(path)])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["report"]["status"], "REFUTED")
        self.assertEqual(payload["report"]["witness"]["constraint"], "initial condition")

    def test_inconclusive_case_returns_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_case(directory, 6)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["verify", str(path)])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["report"]["status"], "INCONCLUSIVE")

    def test_expression_budget_is_reported_as_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_case(directory, 0)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["verify", str(path), "--max-expression-ops", "1"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["report"]["status"], "INCONCLUSIVE")
        self.assertTrue(
            any(
                "configured limit of 1" in reason
                for reason in payload["report"]["incomplete_reasons"].values()
            )
        )

    def test_output_file_receives_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_case(directory, 0)
            report_path = Path(directory) / "report.json"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["verify", str(path), "--output", str(report_path)])
            payload = json.loads(report_path.read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(payload["report"]["status"], "PROVED")

    def test_invalid_case_returns_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{}")
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["verify", str(path)])
        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("missing field", errors.getvalue())

    def test_missing_file_returns_input_error(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(["verify", "does-not-exist.json"])
        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("does-not-exist.json", errors.getvalue())

    def test_template_validate_prints_contract_summary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["template", "validate", "examples/heat-template.json"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["template_version"], 1)
        self.assertEqual(summary["solution_semantics"], "classical_strong")
        self.assertEqual(summary["field_names"], ["u"])
        self.assertEqual(summary["pde_residuals"], 1)
        self.assertEqual(summary["conditions"], 3)

    def test_template_validate_returns_input_error_for_invalid_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-template.json"
            path.write_text("{}")
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["template", "validate", str(path)])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("missing field", errors.getvalue())

    def test_run_validate_checks_the_complete_example_bundle(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["run", "validate", "examples/heat-run-manifest.json"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["manifest_version"], 1)
        self.assertEqual(summary["integrity_scope"], "content_identity_only")
        self.assertEqual(summary["artifact_kind"], "symbolic")
        self.assertEqual(summary["problem_id"], "heat-classical-01")
        self.assertEqual(len(summary["manifest_sha256"]), 64)

    def test_run_validate_returns_input_error_for_tampered_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "heat-template.json",
                "exact-heat-candidate.json",
                "exact-heat-report.json",
                "heat-run-manifest.json",
            ):
                (root / name).write_bytes((Path("examples") / name).read_bytes())
            candidate = root / "exact-heat-candidate.json"
            candidate.write_text(candidate.read_text().replace("sin(pi*x)", "0"))
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["run", "validate", str(root / "heat-run-manifest.json")])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("digest mismatch", errors.getvalue())

    def test_non_finite_tolerance_is_rejected_by_argument_parser(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            main(["verify", "case.json", "--tolerance", "nan"])
        self.assertIn("must be finite and positive", errors.getvalue())

    def test_non_positive_expression_budget_is_rejected_by_argument_parser(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit):
            main(["verify", "case.json", "--max-expression-ops", "0"])
        self.assertIn("must be at least one", errors.getvalue())

    def test_corpus_validate_prints_a_coverage_summary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["corpus", "validate", "corpus/pilot.json"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["corpus_version"], 1)
        self.assertEqual(summary["records"], 20)
        self.assertEqual(summary["annotation_statuses"], {"labeled": 20})
        self.assertEqual(summary["origin_kinds"], {"open_model": 10, "symbolic_solver": 10})
        self.assertEqual(summary["verdicts"], {"invalid": 10, "valid": 10})

    def test_corpus_validate_accepts_a_modular_atlas_directory(self):
        records = [path for path in Path("corpus/community/records").iterdir() if path.is_dir()]
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["corpus", "validate", "corpus/community"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["corpus_version"], 1)
        self.assertEqual(summary["records"], len(records))
        self.assertEqual(summary["name"], "PDE Failure Atlas community intake")
        self.assertEqual(sum(summary["annotation_statuses"].values()), len(records))
        self.assertEqual(sum(summary["origin_kinds"].values()), len(records))
        self.assertGreaterEqual(summary["origin_kinds"]["synthetic"], 6)

    def test_corpus_validate_returns_input_error_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-corpus.json"
            path.write_text("{")
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["corpus", "validate", str(path)])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("invalid JSON", errors.getvalue())

    def test_corpus_validate_reports_modular_atlas_taxonomy(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["corpus", "validate", "corpus/community"])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["coverage_version"], 1)
        self.assertEqual(summary["artifact_types"], {"symbolic_expression": 13})
        self.assertEqual(summary["pde_families"]["heat"], 7)
        self.assertEqual(summary["pde_families"]["poisson"], 1)
        self.assertEqual(summary["spatial_dimensions"], {"1": 11, "2": 2})

    def test_corpus_evaluate_runs_one_symbolic_atlas_record(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "corpus",
                    "evaluate",
                    "corpus/matched",
                    "--record",
                    "qwen3-fisher-kpp-01",
                ]
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["evaluation_version"], 1)
        self.assertEqual(payload["evidence_policy"], "per_record_no_aggregation")
        self.assertEqual(payload["records"][0]["report"]["status"], "PROVED")
        self.assertNotIn("status", payload)

    def test_corpus_evaluate_rejects_an_atlas_v1_source(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(["corpus", "evaluate", "corpus/community"])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("expected 2", errors.getvalue())

    def test_corpus_summarize_evaluation_reads_a_saved_report(self):
        evaluation = evaluate_cross_artifact_atlas(
            "corpus/matched",
            record_ids=["qwen3-fisher-kpp-01"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(evaluation))
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["corpus", "summarize-evaluation", str(path)])

        summary = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["summary_version"], 1)
        self.assertEqual(summary["coverage"]["records"], 1)
        self.assertEqual(summary["outcomes"]["statuses"], {"PROVED": 1})
        self.assertNotIn("status", summary)

    def test_corpus_summarize_evaluation_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{")
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["corpus", "summarize-evaluation", str(path)])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("invalid JSON", errors.getvalue())

    def test_corpus_summarize_evaluation_handles_extreme_numbers_as_input_errors(self):
        evaluation = evaluate_cross_artifact_atlas(
            "corpus/matched",
            record_ids=["qwen3-fisher-kpp-01"],
        )
        evaluation["records"][0]["report"]["max_sampled_residual"] = 10**4_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extreme.json"
            path.write_text(json.dumps(evaluation))
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = main(["corpus", "summarize-evaluation", str(path)])

        self.assertEqual(exit_code, INPUT_ERROR)
        self.assertIn("report", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
