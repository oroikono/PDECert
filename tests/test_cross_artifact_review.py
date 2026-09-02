import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.apply_review import main as apply_review_main
from experiments.review_corpus import (
    load_or_create_review,
    new_review,
    render_record,
    run_session,
)
from pdecert import (
    ReviewError,
    apply_review,
    load_cross_artifact_atlas,
    review_source_sha256,
)


ATLAS = Path("corpus/matched")
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SYMBOLIC_ID = "qwen3-fisher-kpp-01"


def _unclear_review(atlas):
    return {
        "atlas_sha256": review_source_sha256(atlas),
        "records": [
            {
                "basis": {
                    "description": "The test reviewer found the represented scope insufficient.",
                    "kind": "scope_assessment",
                },
                "failure_modes": [],
                "id": record["id"],
                "rationale": "The test review leaves this fixture unclear.",
                "verdict": "unclear",
            }
            for record in atlas["records"]
        ],
        "review_version": 2,
    }


def _non_annotation_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "record.json"
    }


class CrossArtifactReviewTests(unittest.TestCase):
    def test_new_review_is_digest_bound_and_keeps_every_decision_blank(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        review = new_review(atlas)

        self.assertEqual(review["review_version"], 2)
        self.assertEqual(review["atlas_sha256"], review_source_sha256(atlas))
        self.assertEqual(
            [record["id"] for record in review["records"]],
            [record["id"] for record in atlas["records"]],
        )
        self.assertTrue(
            all(
                record["verdict"] is None
                and record["rationale"] is None
                and record["basis"] is None
                for record in review["records"]
            )
        )

    def test_callable_card_omits_training_and_machine_outcomes(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        record = next(record for record in atlas["records"] if record["id"] == CALLABLE_ID)
        rendered = render_record(record, 1, 1)

        self.assertIn("Artifact type: callable_model", rendered)
        self.assertIn("Weights SHA-256", rendered)
        self.assertNotIn("final_losses", rendered)
        self.assertNotIn("PDECert", rendered)
        self.assertNotIn("REFUTED", rendered)
        self.assertNotIn("PROVED", rendered)

    def test_typed_session_records_scope_basis_for_an_unclear_decision(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        atlas = {**atlas, "records": atlas["records"][:1]}
        review = new_review(atlas)
        answers = iter(
            [
                "u",
                "1",
                "The represented semantics do not establish the missing regularity assumption.",
                "The problem scope is insufficient for an independent binary decision.",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            complete = run_session(
                atlas,
                review,
                output,
                input_fn=lambda prompt: next(answers),
                output_fn=lambda message: None,
            )
            resumed = load_or_create_review(output, atlas)

        self.assertTrue(complete)
        self.assertEqual(resumed["records"][0]["verdict"], "unclear")
        self.assertEqual(resumed["records"][0]["basis"]["kind"], "scope_assessment")

    def test_review_digest_must_match_the_exact_loaded_atlas(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        review = _unclear_review(atlas)
        review["atlas_sha256"] = "0" * 64

        with self.assertRaisesRegex(ReviewError, "does not match"):
            apply_review(
                atlas,
                review,
                annotator="test-reviewer",
                confirmed_independent_review=True,
            )

    def test_callable_valid_label_cannot_rest_on_manual_derivation(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        review = _unclear_review(atlas)
        decision = next(record for record in review["records"] if record["id"] == CALLABLE_ID)
        decision.update(
            verdict="valid",
            rationale="A deliberately unsupported test rationale.",
            basis={
                "description": "The reviewer only inspected the model architecture.",
                "kind": "manual_derivation",
            },
        )

        with self.assertRaisesRegex(ReviewError, "cannot support.*callable_model"):
            apply_review(
                atlas,
                review,
                annotator="test-reviewer",
                confirmed_independent_review=True,
            )

    def test_completed_review_is_imported_without_changing_artifact_bytes(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        review = _unclear_review(atlas)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "review.json"
            output = root / "labeled-atlas"
            review_path.write_text(json.dumps(review))
            before = _non_annotation_hashes(ATLAS)

            apply_review_main(
                [
                    str(review_path),
                    "--corpus",
                    str(ATLAS),
                    "--output",
                    str(output),
                    "--annotator",
                    "test-reviewer",
                    "--confirm-independent-review",
                ]
            )

            labeled = load_cross_artifact_atlas(output)
            after = _non_annotation_hashes(output)

        self.assertEqual(before, after)
        self.assertTrue(
            all(record["annotation"]["status"] == "labeled" for record in labeled["records"])
        )
        self.assertTrue(
            all(
                record["annotation"]["review_basis"]["kind"] == "scope_assessment"
                for record in labeled["records"]
            )
        )
        unchanged = load_cross_artifact_atlas(ATLAS)
        self.assertTrue(
            all(record["annotation"]["status"] == "pending" for record in unchanged["records"])
        )

    def test_apply_review_does_not_mutate_loaded_source(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        original = copy.deepcopy(atlas)
        apply_review(
            atlas,
            _unclear_review(atlas),
            annotator="test-reviewer",
            confirmed_independent_review=True,
        )
        self.assertEqual(atlas, original)


if __name__ == "__main__":
    unittest.main()
