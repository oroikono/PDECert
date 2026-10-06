import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.apply_review import main as apply_review_main
from pdecert import (
    ReviewError,
    apply_review,
    corpus_sha256,
    dump_atlas,
    load_atlas,
    load_atlas_coverage,
    load_corpus,
    validate_corpus,
)


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

    def _review_command_files(self, root):
        source = copy.deepcopy(self.corpus)
        source["records"] = source["records"][:1]
        review = {
            "review_version": 1,
            "records": [
                {
                    "id": source["records"][0]["id"],
                    "verdict": "unclear",
                    "failure_modes": [],
                    "rationale": "Synthetic import test, not a scientific judgment.",
                }
            ],
        }
        source_path, review_path = root / "source.json", root / "review.json"
        source_path.write_text(json.dumps(source))
        review_path.write_text(json.dumps(review))
        return source_path, review_path

    def _run_review_command(self, source, review, output, *, confirmed=True):
        arguments = [
            str(review),
            "--corpus",
            str(source),
            "--output",
            str(output),
            "--annotator",
            "synthetic-test-reviewer",
        ]
        if confirmed:
            arguments.append("--confirm-independent-review")
        apply_review_main(arguments)

    def test_review_command_writes_a_complete_new_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._review_command_files(root)
            original = {path: path.read_bytes() for path in (source, review)}
            output = root / "labeled.json"
            self._run_review_command(source, review, output)
            labeled = load_corpus(output)
            self.assertEqual(labeled["records"][0]["annotation"]["verdict"], "unclear")
            self.assertEqual(
                labeled["records"][0]["annotation"]["annotators"], ["synthetic-test-reviewer"]
            )
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(set(root.iterdir()), {source, review, output})

    def test_review_command_refuses_existing_json_destinations_and_aliases(self):
        targets = ("source", "review", "unrelated", "directory", "symlink", "hardlink", "dangling")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, review = self._review_command_files(root)
                output = root / "labeled.json"
                if target in {"source", "review"}:
                    output = source if target == "source" else review
                elif target == "unrelated":
                    output.write_bytes(b"previous work\n")
                elif target == "directory":
                    output.mkdir()
                elif target == "symlink":
                    output.symlink_to(review)
                elif target == "hardlink":
                    os.link(source, output)
                else:
                    output.symlink_to(root / "absent.json")
                original = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
                entries = set(root.iterdir())
                with self.assertRaisesRegex(ReviewError, "refusing to overwrite"):
                    self._run_review_command(source, review, output)
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content)
                self.assertEqual(set(root.iterdir()), entries)
                if target == "dangling":
                    self.assertTrue(output.is_symlink())
                    self.assertFalse((root / "absent.json").exists())

    def test_review_command_does_not_publish_unconfirmed_or_invalid_review(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, review = self._review_command_files(root)
                if invalid:
                    data = json.loads(review.read_text())
                    data["records"][0]["rationale"] = ""
                    review.write_text(json.dumps(data))
                original = {path: path.read_bytes() for path in (source, review)}
                with self.assertRaises(ReviewError):
                    self._run_review_command(
                        source, review, root / "labeled.json", confirmed=invalid
                    )
                self.assertEqual(set(root.iterdir()), {source, review})
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_review_command_does_not_publish_a_partial_json_write(self):
        def fail_write(value, path):
            Path(path).write_text('{"partial":')
            raise OSError("synthetic disk-full error")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._review_command_files(root)
            with patch("experiments.apply_review.dump_corpus", side_effect=fail_write):
                with self.assertRaisesRegex(OSError, "disk-full"):
                    self._run_review_command(source, review, root / "labeled.json")
            self.assertEqual(set(root.iterdir()), {source, review})

    def test_review_command_preserves_a_destination_created_during_publication(self):
        def competing_writer(staged, destination):
            # The complete file must be staged before the destination becomes visible.
            load_corpus(staged)
            Path(destination).write_bytes(b"another review\n")
            raise FileExistsError("synthetic concurrent creation")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._review_command_files(root)
            output = root / "labeled.json"
            with patch("experiments.apply_review.os.link", side_effect=competing_writer):
                with self.assertRaisesRegex(ReviewError, "refusing to overwrite"):
                    self._run_review_command(source, review, output)
            self.assertEqual(output.read_bytes(), b"another review\n")
            self.assertEqual(set(root.iterdir()), {source, review, output})

    def test_review_command_has_no_overwrite_fallback_when_links_are_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, review = self._review_command_files(root)
            with patch("experiments.apply_review.os.link", side_effect=OSError("unsupported link")):
                with self.assertRaisesRegex(OSError, "unsupported link"):
                    self._run_review_command(source, review, root / "labeled.json")
            self.assertEqual(set(root.iterdir()), {source, review})

    def test_review_command_writes_a_new_modular_atlas(self):
        source = copy.deepcopy(self.corpus)
        review = {
            "review_version": 1,
            "records": [
                {
                    "failure_modes": [],
                    "id": record["id"],
                    "rationale": "The test reviewer found the written semantics unclear.",
                    "verdict": "unclear",
                }
                for record in source["records"]
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "pending-atlas"
            review_path = root / "review.json"
            output = root / "labeled-atlas"
            coverage = {
                "coverage_version": 1,
                "records": {
                    record["id"]: {
                        "artifact_type": "symbolic_expression",
                        "pde_families": ["test_family"],
                        "spatial_dimension": 1,
                    }
                    for record in source["records"]
                },
            }
            dump_atlas(source, source_path, coverage=coverage)
            review_path.write_text(json.dumps(review))
            apply_review_main(
                [
                    str(review_path),
                    "--corpus",
                    str(source_path),
                    "--output",
                    str(output),
                    "--annotator",
                    "test-reviewer",
                    "--confirm-independent-review",
                ]
            )
            labeled = load_atlas(output)
            unchanged = load_atlas(source_path)
            labeled_coverage = load_atlas_coverage(
                output,
                {record["id"] for record in labeled["records"]},
            )

        self.assertTrue(
            all(record["annotation"]["status"] == "pending" for record in unchanged["records"])
        )
        self.assertTrue(
            all(record["annotation"]["status"] == "labeled" for record in labeled["records"])
        )
        self.assertTrue(
            all(
                record["annotation"]["annotators"] == ["test-reviewer"]
                for record in labeled["records"]
            )
        )
        self.assertEqual(labeled_coverage, coverage)


if __name__ == "__main__":
    unittest.main()
