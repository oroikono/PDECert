import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from experiments.adversarial_heat import build_cases
from pdecert import VerificationCase, dump_case
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
        self.assertEqual(payload["report"]["decision_evidence"], "EXACT")
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


if __name__ == "__main__":
    unittest.main()
