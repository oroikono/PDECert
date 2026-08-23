import copy
import hashlib
import json
import unittest
from pathlib import Path

from pdecert import ReviewError, apply_review, corpus_sha256, load_corpus, validate_corpus


class LabelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.committed_corpus = load_corpus("corpus/pilot.json")
        cls.corpus = copy.deepcopy(cls.committed_corpus)
        for record in cls.corpus["records"]:
            record["annotation"] = {
                "annotators": [],
                "failure_modes": [],
                "rationale": None,
                "status": "pending",
                "verdict": None,
            }
        cls.provisional = json.loads(Path("results/provisional-review.json").read_text())

    def test_committed_corpus_is_fully_human_labeled(self):
        self.assertTrue(
            all(
                record["annotation"]["status"] == "labeled"
                and record["annotation"]["annotators"] == ["oroikono"]
                for record in self.committed_corpus["records"]
            )
        )

    def test_comparison_note_matches_committed_annotations(self):
        comparison = json.loads(Path("corpus/review-comparison.json").read_text())
        review = {
            "review_version": 1,
            "records": [
                {
                    "failure_modes": record["annotation"]["failure_modes"],
                    "id": record["id"],
                    "rationale": record["annotation"]["rationale"],
                    "verdict": record["annotation"]["verdict"],
                }
                for record in self.committed_corpus["records"]
            ],
        }
        encoded = json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            comparison["amended_review_sha256"],
        )
        self.assertEqual(
            corpus_sha256(self.committed_corpus),
            comparison["labeled_corpus_sha256"],
        )
        self.assertEqual(comparison["confirmed_by"], "oroikono")
        self.assertEqual(comparison["provisional_comparison"]["verdict_disagreements"], 0)

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
