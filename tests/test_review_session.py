import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.review_corpus import (
    ReviewSessionError,
    load_or_create_review,
    new_review,
    progress_bar,
    render_record,
    run_session,
)
from pdecert import load_corpus


def _one_record_corpus():
    corpus = copy.deepcopy(load_corpus("corpus/pilot.json"))
    corpus["records"] = corpus["records"][:1]
    return corpus


class ReviewSessionTests(unittest.TestCase):
    def test_card_contains_source_material_but_no_answer(self):
        corpus = _one_record_corpus()
        rendered = render_record(corpus["records"][0], 1, 1)
        self.assertIn("Candidate fields", rendered)
        self.assertIn("PDE residuals", rendered)
        self.assertIn("Unedited generator output", rendered)
        self.assertNotIn(corpus["records"][0]["origin"]["producer"], rendered)
        self.assertNotIn("PROVED", rendered)
        self.assertNotIn("provisional", rendered.lower())

    def test_valid_decision_is_saved_and_completes_session(self):
        corpus = _one_record_corpus()
        review = new_review(corpus)
        answers = iter(["v", "Direct substitution makes every represented residual zero."])
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            complete = run_session(
                corpus,
                review,
                output,
                input_fn=lambda prompt: next(answers),
                output_fn=messages.append,
            )
            resumed = load_or_create_review(output, corpus)
        self.assertTrue(complete)
        self.assertEqual(resumed["records"][0]["verdict"], "valid")
        self.assertEqual(resumed["records"][0]["failure_modes"], [])
        self.assertTrue(any("1/1" in message for message in messages))

    def test_invalid_decision_requires_and_saves_failure_modes(self):
        corpus = _one_record_corpus()
        review = new_review(corpus)
        answers = iter(["i", "7,4", "The PDE and initial trace are nonzero."])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            complete = run_session(
                corpus,
                review,
                output,
                input_fn=lambda prompt: next(answers),
                output_fn=lambda message: None,
            )
        self.assertTrue(complete)
        self.assertEqual(
            review["records"][0]["failure_modes"],
            ["pde_residual", "initial_condition"],
        )

    def test_quit_saves_resumable_blank_review(self):
        corpus = _one_record_corpus()
        review = new_review(corpus)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            complete = run_session(
                corpus,
                review,
                output,
                input_fn=lambda prompt: "q",
                output_fn=lambda message: None,
            )
            resumed = load_or_create_review(output, corpus)
        self.assertFalse(complete)
        self.assertIsNone(resumed["records"][0]["verdict"])

    def test_interruption_during_rationale_preserves_resumable_review(self):
        corpus = _one_record_corpus()
        review = new_review(corpus)

        def interrupted_input(prompt):
            if "Verdict" in prompt:
                return "v"
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            complete = run_session(
                corpus,
                review,
                output,
                input_fn=interrupted_input,
                output_fn=lambda message: None,
            )
            resumed = load_or_create_review(output, corpus)
        self.assertFalse(complete)
        self.assertIsNone(resumed["records"][0]["verdict"])

    def test_resume_refuses_different_corpus_order(self):
        corpus = copy.deepcopy(load_corpus("corpus/pilot.json"))
        review = new_review(corpus)
        review["records"].reverse()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            output.write_text(json.dumps(review))
            with self.assertRaisesRegex(ReviewSessionError, "do not match"):
                load_or_create_review(output, corpus)

    def test_progress_bar_reaches_full_width(self):
        self.assertEqual(progress_bar(20, 20), "[####################] 20/20")


if __name__ == "__main__":
    unittest.main()
