import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.audit_benchmark_coverage import audit_benchmark_coverage, main
from pdecert import corpus_sha256, dump_corpus, load_corpus


class BenchmarkCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = load_corpus("corpus/pilot.json")

    def audit_modified_pilot(self, records):
        corpus = copy.deepcopy(self.pilot)
        corpus["records"] = records
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.json"
            dump_corpus(corpus, source)
            return audit_benchmark_coverage(source)

    def test_pilot_exposes_confounded_origins_and_single_annotator(self):
        with patch(
            "pdecert.benchmark.evaluate_corpus", side_effect=AssertionError("no evaluation")
        ):
            report = audit_benchmark_coverage("corpus/pilot.json")
        self.assertEqual(report["corpus"]["record_count"], 20)
        self.assertEqual(report["corpus"]["sha256"], corpus_sha256(self.pilot))
        self.assertEqual(report["origin_grouping"], "origin.kind")
        rows = {row["origin_kind"]: row for row in report["origin_verdict_counts"]}
        self.assertEqual(
            rows["symbolic_solver"]["verdict_counts"],
            {"valid": 10, "invalid": 0, "unclear": 0, "pending": 0},
        )
        self.assertEqual(
            rows["open_model"]["verdict_counts"],
            {"valid": 0, "invalid": 10, "unclear": 0, "pending": 0},
        )
        self.assertEqual(sum(row["record_count"] for row in rows.values()), 20)
        self.assertEqual(rows["open_model"]["producer_identifiers"][0]["record_count"], 10)
        diagnostic = report["origin_verdict_diagnostic"]
        self.assertTrue(diagnostic["origin_determines_binary_verdict"])
        self.assertEqual(diagnostic["binary_record_count"], 20)
        self.assertEqual(diagnostic["excluded_pending_or_unclear_record_count"], 0)
        reviewers = report["reviewer_coverage"]
        self.assertEqual(reviewers["unique_annotator_id_count"], 1)
        self.assertEqual(len(reviewers["record_ids_by_annotator"]["oroikono"]), 20)
        self.assertEqual(
            reviewers["record_counts_by_annotator_id_count"],
            {"zero": 0, "one": 20, "two_or_more": 0},
        )
        self.assertIsNone(reviewers["independent_reviewer_count"])

    def test_mixed_verdicts_within_each_origin_are_retained(self):
        records = copy.deepcopy(self.pilot["records"])
        valid = next(record for record in records if record["annotation"]["verdict"] == "valid")
        invalid = next(record for record in records if record["annotation"]["verdict"] == "invalid")
        valid["annotation"], invalid["annotation"] = invalid["annotation"], valid["annotation"]
        report = self.audit_modified_pilot(records)
        diagnostic = report["origin_verdict_diagnostic"]
        self.assertFalse(diagnostic["origin_determines_binary_verdict"])
        self.assertEqual(
            diagnostic["origins_with_both_binary_verdicts"], ["open_model", "symbolic_solver"]
        )
        self.assertEqual(diagnostic["single_binary_verdict_by_origin"], {})
        rows = {
            row["origin_kind"]: row["verdict_counts"] for row in report["origin_verdict_counts"]
        }
        self.assertEqual(rows["symbolic_solver"]["invalid"], 1)
        self.assertEqual(rows["open_model"]["valid"], 1)

    def test_pending_and_unclear_have_separate_denominators(self):
        records = copy.deepcopy(self.pilot["records"])
        records[0]["annotation"] = {
            "annotators": [],
            "failure_modes": [],
            "rationale": None,
            "status": "pending",
            "verdict": None,
        }
        records[1]["annotation"].update(verdict="unclear", failure_modes=[])
        report = self.audit_modified_pilot(records)
        coverage = report["annotation_coverage"]
        self.assertEqual(coverage["pending_record_count"], 1)
        self.assertEqual(coverage["completed_record_count"], 19)
        self.assertEqual(report["origin_verdict_diagnostic"]["binary_record_count"], 18)
        self.assertEqual(
            report["origin_verdict_diagnostic"]["excluded_pending_or_unclear_record_count"], 2
        )
        rows = {
            row["origin_kind"]: row["verdict_counts"] for row in report["origin_verdict_counts"]
        }
        self.assertEqual(rows["symbolic_solver"]["pending"], 1)
        self.assertEqual(rows["symbolic_solver"]["unclear"], 1)

    def test_multiple_ids_and_adjudication_do_not_establish_independence(self):
        records = copy.deepcopy(self.pilot["records"][:2])
        records[0]["annotation"].update(
            annotators=["reviewer-a", "reviewer-b"], status="adjudicated"
        )
        records[1]["annotation"].update(annotators=["reviewer-b"])
        report = self.audit_modified_pilot(records)
        reviewers = report["reviewer_coverage"]
        self.assertEqual(reviewers["unique_annotator_id_count"], 2)
        self.assertEqual(reviewers["record_counts_by_annotator_id_count"]["two_or_more"], 1)
        self.assertEqual(len(reviewers["record_ids_by_annotator"]["reviewer-b"]), 2)
        self.assertEqual(
            reviewers["independent_review_status"], "not_established_by_annotation_metadata"
        )
        self.assertIsNone(reviewers["independent_reviewer_count"])
        self.assertEqual(report["annotation_coverage"]["status_counts"]["adjudicated"], 1)
        self.assertIsNone(report["origin_verdict_diagnostic"]["origin_determines_binary_verdict"])

    def test_empty_corpus_has_no_separation_or_independence_claim(self):
        report = self.audit_modified_pilot([])
        self.assertEqual(report["corpus"]["record_count"], 0)
        self.assertEqual(report["origin_verdict_counts"], [])
        self.assertIsNone(report["origin_verdict_diagnostic"]["origin_determines_binary_verdict"])
        self.assertIsNone(report["reviewer_coverage"]["independent_reviewer_count"])
        json.dumps(report, allow_nan=False)

    def test_modular_and_typed_atlases_are_pending_without_evaluation(self):
        for source, count, digest_kind in (
            ("corpus/community", 13, "canonical_loaded_corpus"),
            ("corpus/matched", 2, "canonical_loaded_atlas_v2"),
        ):
            with self.subTest(source=source):
                report = audit_benchmark_coverage(source)
                self.assertEqual(report["corpus"]["digest_kind"], digest_kind)
                self.assertEqual(report["corpus"]["record_count"], count)
                self.assertEqual(report["annotation_coverage"]["pending_record_count"], count)
                self.assertEqual(report["reviewer_coverage"]["unique_annotator_id_count"], 0)
                self.assertIsNone(
                    report["origin_verdict_diagnostic"]["origin_determines_binary_verdict"]
                )

    def test_command_is_deterministic_json_and_does_not_modify_source(self):
        source = Path("corpus/pilot.json")
        before = source.read_bytes()
        outputs = []
        for _ in range(2):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([]), 0)
            outputs.append(output.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(json.loads(outputs[0])["scope"], "descriptive_stored_annotations_only")
        self.assertEqual(source.read_bytes(), before)

    def test_command_rejects_invalid_or_unsupported_sources_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "corpus.json"
            for payload in (
                "not JSON",
                json.dumps({**self.pilot, "corpus_version": 99}),
                json.dumps({**self.pilot, "records": [{"id": "missing-fields"}]}),
            ):
                source.write_text(payload)
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(main([str(source)]), 2)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("audit_benchmark_coverage:", errors.getvalue())
                self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
