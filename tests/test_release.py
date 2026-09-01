import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pdecert import (
    ReleaseError,
    build_release_bundle,
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


def _tree_contents(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labeled = load_corpus("corpus/pilot.json")
        cls.pending = _pending_fixture(cls.labeled)
        cls.benchmark = evaluate_corpus(cls.labeled, symbolic_timeout=1.0)

    def test_pending_corpus_is_refused_before_release(self):
        with self.assertRaisesRegex(ReleaseError, "completed human labels"):
            validate_release_inputs(self.pending, {})

    def test_benchmark_must_match_exact_corpus_digest(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["corpus"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReleaseError, "digest does not match"):
            validate_release_inputs(self.labeled, benchmark)

    def test_tampered_metrics_are_refused(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["methods"]["pdecert"]["metrics"]["correct_count"] -= 1
        with self.assertRaisesRegex(ReleaseError, "metrics do not match"):
            validate_release_inputs(self.labeled, benchmark)

    def test_boolean_version_is_not_treated_as_integer_one(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["benchmark_version"] = True
        with self.assertRaisesRegex(ReleaseError, "benchmark_version"):
            validate_release_inputs(self.labeled, benchmark)

    def test_tampered_truth_is_refused(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["methods"]["pdecert"]["records"][0]["truth"] = "invalid"
        with self.assertRaisesRegex(ReleaseError, "truth does not match"):
            validate_release_inputs(self.labeled, benchmark)

    def test_release_bundle_is_viewer_ready_and_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            manifest = build_release_bundle(self.labeled, self.benchmark, output)
            contents = _tree_contents(output)

        self.assertEqual(
            set(contents),
            {
                "README.md",
                "data/pilot.jsonl",
                "manifest.json",
                "results/pilot-benchmark.json",
            },
        )
        rows = [json.loads(line) for line in contents["data/pilot.jsonl"].splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["annotation"]["status"] == "labeled" for row in rows))
        card = contents["README.md"].decode()
        self.assertTrue(card.startswith("---\nlicense: other\n"))
        self.assertIn("path: data/pilot.jsonl", card)
        self.assertIn("not a random or representative sample", card)
        self.assertIn("does not by itself settle every right", card)
        self.assertIn("frozen symbolic-only pilot", card)
        self.assertIn("LIMITATIONS_AND_THREATS_TO_VALIDITY.md", card)
        self.assertIn("| PDECert |", card)
        self.assertEqual(manifest["record_count"], 20)
        self.assertEqual(manifest["verdict_counts"], {"invalid": 10, "valid": 10})
        for name, expected_digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256(contents[name]).hexdigest(), expected_digest)
        json.loads(contents["results/pilot-benchmark.json"])
        json.loads(contents["manifest.json"])

    def test_same_inputs_build_byte_identical_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            build_release_bundle(self.labeled, self.benchmark, first)
            build_release_bundle(self.labeled, self.benchmark, second)
            self.assertEqual(_tree_contents(first), _tree_contents(second))

    def test_nonempty_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            output.mkdir()
            (output / "keep.txt").write_text("do not overwrite")
            with self.assertRaisesRegex(ReleaseError, "not empty"):
                build_release_bundle(self.labeled, self.benchmark, output)
            self.assertEqual((output / "keep.txt").read_text(), "do not overwrite")

    def test_existing_file_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            output.write_text("do not overwrite")
            with self.assertRaisesRegex(ReleaseError, "not a directory"):
                build_release_bundle(self.labeled, self.benchmark, output)
            self.assertEqual(output.read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
