import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from experiments.run_benchmark import main as benchmark_main
from pdecert import (
    BenchmarkError,
    dump_corpus,
    evaluate_corpus,
    load_corpus,
    validate_release_inputs,
)


def _pending_fixture(corpus):
    pending = copy.deepcopy(corpus)
    for record in pending["records"]:
        record["annotation"] = {
            "annotators": [],
            "failure_modes": [],
            "rationale": None,
            "status": "pending",
            "verdict": None,
        }
    return pending


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labeled = load_corpus("corpus/pilot.json")
        cls.pending = _pending_fixture(cls.labeled)

    def test_pending_labels_are_refused(self):
        with self.assertRaisesRegex(BenchmarkError, "completed human labels"):
            evaluate_corpus(self.pending)

    def test_three_methods_return_comparable_metrics(self):
        report = evaluate_corpus(self.labeled, symbolic_timeout=1.0)
        self.assertEqual(report["benchmark_version"], 1)
        self.assertEqual(report["corpus"]["scored_records"], 20)
        self.assertEqual(len(report["corpus"]["sha256"]), 64)
        self.assertEqual(
            set(report["environment"]),
            {"pdecert", "platform", "python", "sympy"},
        )
        self.assertEqual(set(report["method_definitions"]), set(report["methods"]))
        self.assertEqual(
            set(report["methods"]),
            {"fixed_collocation", "pdecert", "sympy_residual"},
        )
        for result in report["methods"].values():
            self.assertEqual(len(result["records"]), 20)
            self.assertEqual(result["metrics"]["valid_count"], 10)
            self.assertEqual(result["metrics"]["invalid_count"], 10)
            self.assertGreaterEqual(result["runtime_seconds"], 0)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["methods"]["pdecert"]["metrics"]["correct_count"], 20)
        self.assertEqual(report["methods"]["pdecert"]["metrics"]["invalid_witness_count"], 10)

    def test_committed_report_is_bound_to_the_labeled_corpus(self):
        report = json.loads(Path("results/pilot-benchmark.json").read_text())
        validate_release_inputs(self.labeled, report)
        self.assertEqual(report["methods"]["fixed_collocation"]["metrics"]["accuracy"], 1.0)
        self.assertEqual(report["methods"]["pdecert"]["metrics"]["accuracy"], 1.0)
        self.assertEqual(
            report["methods"]["pdecert"]["metrics"]["invalid_witness_rate"],
            1.0,
        )
        self.assertEqual(
            report["methods"]["sympy_residual"]["metrics"]["inconclusive_rate"],
            0.35,
        )

    def test_unclear_records_are_reported_and_excluded(self):
        corpus = copy.deepcopy(self.labeled)
        corpus["records"][0]["annotation"] = {
            "annotators": ["oroikono"],
            "failure_modes": [],
            "rationale": "The problem statement is insufficient for a decision.",
            "status": "labeled",
            "verdict": "unclear",
        }
        report = evaluate_corpus(corpus, symbolic_timeout=1.0)
        self.assertEqual(report["corpus"]["excluded_unclear"], 1)
        self.assertEqual(report["corpus"]["scored_records"], 19)

    def test_invalid_controls_are_rejected(self):
        with self.assertRaisesRegex(BenchmarkError, "points_per_axis"):
            evaluate_corpus(self.labeled, points_per_axis=1)
        with self.assertRaisesRegex(BenchmarkError, "tolerance"):
            evaluate_corpus(self.labeled, tolerance=0)
        with self.assertRaisesRegex(BenchmarkError, "symbolic_timeout"):
            evaluate_corpus(self.labeled, symbolic_timeout=0)

    def test_command_refuses_pending_corpus_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "pending.json"
            dump_corpus(self.pending, corpus_path)
            output = Path(directory) / "report.json"
            errors = io.StringIO()
            with redirect_stderr(errors):
                exit_code = benchmark_main([str(corpus_path), "--output", str(output)])
            self.assertEqual(exit_code, 2)
            self.assertIn("completed human labels", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
