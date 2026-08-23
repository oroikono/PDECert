import copy
import json
import unittest
from pathlib import Path

from pdecert import ReviewError, apply_review, load_corpus, validate_corpus


class LabelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus("corpus/pilot.json")
        cls.provisional = json.loads(Path("results/provisional-review.json").read_text())

    def test_independent_review_confirmation_is_required(self):
        with self.assertRaisesRegex(ReviewError, "independent human review"):
            apply_review(
                self.corpus,
                self.provisional,
                annotator="reviewer",
                confirmed_independent_review=False,
            )

    def test_complete_review_can_be_applied_without_mutating_source(self):
        source = copy.deepcopy(self.corpus)
        labeled = apply_review(
            self.corpus,
            self.provisional,
            annotator="reviewer",
            confirmed_independent_review=True,
        )
        validate_corpus(labeled)
        self.assertEqual(self.corpus, source)
        self.assertTrue(
            all(record["annotation"]["status"] == "labeled" for record in labeled["records"])
        )
        self.assertTrue(
            all(record["annotation"]["annotators"] == ["reviewer"] for record in labeled["records"])
        )
        verdicts = [record["annotation"]["verdict"] for record in labeled["records"]]
        self.assertEqual(verdicts.count("valid"), 10)
        self.assertEqual(verdicts.count("invalid"), 10)

    def test_review_ids_must_match_corpus(self):
        review = copy.deepcopy(self.provisional)
        review["records"].pop()
        with self.assertRaisesRegex(ReviewError, "record IDs do not match corpus"):
            apply_review(
                self.corpus,
                review,
                annotator="reviewer",
                confirmed_independent_review=True,
            )

    def test_blank_template_cannot_be_applied(self):
        template = json.loads(Path("corpus/review-template.json").read_text())
        with self.assertRaisesRegex(ReviewError, "expected one of"):
            apply_review(
                self.corpus,
                template,
                annotator="reviewer",
                confirmed_independent_review=True,
            )

    def test_invalid_verdict_requires_failure_mode(self):
        review = copy.deepcopy(self.provisional)
        decision = next(item for item in review["records"] if item["verdict"] == "invalid")
        decision["failure_modes"] = []
        with self.assertRaisesRegex(ReviewError, "require a failure mode"):
            apply_review(
                self.corpus,
                review,
                annotator="reviewer",
                confirmed_independent_review=True,
            )

    def test_existing_annotations_are_not_overwritten(self):
        labeled = apply_review(
            self.corpus,
            self.provisional,
            annotator="reviewer",
            confirmed_independent_review=True,
        )
        with self.assertRaisesRegex(ReviewError, "already annotated"):
            apply_review(
                labeled,
                self.provisional,
                annotator="second-reviewer",
                confirmed_independent_review=True,
            )


if __name__ == "__main__":
    unittest.main()
