import json
import unittest
from pathlib import Path

from experiments.collect_pilot import (
    _extract_expression,
    _model_problems,
    _prompt,
    collect_sympy_records,
)
from pdecert import load_corpus, validate_corpus


class PilotCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sympy_records = collect_sympy_records()

    def test_sympy_collection_produces_ten_real_solver_records(self):
        self.assertEqual(len(self.sympy_records), 10)
        self.assertEqual(len({record["id"] for record in self.sympy_records}), 10)
        self.assertTrue(
            all(record["raw_output"].startswith("Eq(u(x, t),") for record in self.sympy_records)
        )
        validate_corpus(
            {
                "corpus_version": 1,
                "name": "solver test",
                "description": "Ten outputs collected from live SymPy pdsolve calls.",
                "records": self.sympy_records,
            }
        )

    def test_open_model_problem_set_has_ten_unique_inputs(self):
        problems = _model_problems()
        self.assertEqual(len(problems), 10)
        self.assertEqual(len({problem.record_id for problem in problems}), 10)
        self.assertEqual(len({_prompt(problem) for problem in problems}), 10)

    def test_expression_extraction_preserves_model_text(self):
        content = "FINAL: exp(-pi**2*t)*sin(pi*x)"
        self.assertEqual(_extract_expression(content), "exp(-pi**2*t)*sin(pi*x)")

    def test_expression_extraction_normalizes_common_surface_syntax(self):
        content = "FINAL: u(x,t) = sin(pi*x) * exp(-x^2)"
        self.assertEqual(_extract_expression(content), "sin(pi*x) * exp(-x**2)")
        self.assertEqual(_extract_expression("FINAL: sin(2πx) * exp(-t)"), "sin(2*pi*x) * exp(-t)")
        self.assertEqual(
            _extract_expression("u = sin(pi*x) + exp(-pi*x)**2"),
            "sin(pi*x) + exp(-pi*x)**2",
        )

    def test_expression_extraction_rejects_nonconforming_output(self):
        with self.assertRaisesRegex(ValueError, "does not contain one extractable"):
            _extract_expression("Here is the answer:\nsin(pi*x)")

    def test_committed_pilot_contains_twenty_pending_real_outputs(self):
        corpus = load_corpus("corpus/pilot.json")
        self.assertEqual(len(corpus["records"]), 20)
        solver_records = [
            record for record in corpus["records"] if record["origin"]["kind"] == "symbolic_solver"
        ]
        model_records = [
            record for record in corpus["records"] if record["origin"]["kind"] == "open_model"
        ]
        self.assertEqual(len(solver_records), 10)
        self.assertEqual(len(model_records), 10)
        self.assertTrue(
            all(record["annotation"]["status"] == "pending" for record in corpus["records"])
        )
        self.assertGreaterEqual(len({record["case"]["fields"]["u"] for record in model_records}), 8)

        raw_directory = Path("corpus/raw")
        self.assertEqual(len(list(raw_directory.glob("*.txt"))), 10)
        for record in model_records:
            transcript = (raw_directory / f"{record['id']}.txt").read_text()
            self.assertEqual(transcript, record["raw_output"])
            responses = json.loads(transcript)["responses"]
            self.assertGreaterEqual(len(responses), 1)
            self.assertLessEqual(len(responses), 3)


if __name__ == "__main__":
    unittest.main()
