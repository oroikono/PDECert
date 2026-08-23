import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.coupled_wave import build_case
from pdecert import (
    CorpusError,
    case_to_dict,
    dump_corpus,
    load_corpus,
    output_sha256,
    validate_corpus,
)


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_output = "u = sin(pi*x)*cos(pi*t); v = cos(pi*x)*sin(pi*t)"
        cls.record = {
            "id": "sympy-wave-001",
            "case": case_to_dict(build_case()),
            "origin": {
                "kind": "symbolic_solver",
                "producer": "SymPy",
                "version": "1.14.0",
                "identifier": "sympy.solvers.pde.pdsolve",
                "revision": None,
                "source_url": "https://docs.sympy.org/latest/modules/solvers/pde.html",
                "license": "BSD-3-Clause",
                "generated_at": "2026-08-23T20:00:00+02:00",
                "input": "Solve the declared coupled first-order wave system.",
            },
            "raw_output": raw_output,
            "output_sha256": output_sha256(raw_output),
            "annotation": {
                "status": "pending",
                "verdict": None,
                "failure_modes": [],
                "rationale": None,
                "annotators": [],
            },
        }
        cls.corpus = {
            "corpus_version": 1,
            "name": "test corpus",
            "description": "A test corpus with one provenance-bearing record.",
            "records": [cls.record],
        }

    def test_valid_record_passes(self):
        validate_corpus(self.corpus)

    def test_file_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            dump_corpus(self.corpus, path)
            loaded = load_corpus(path)
            self.assertTrue(path.read_text().endswith("\n"))
        self.assertEqual(loaded, self.corpus)

    def test_digest_must_match_raw_output(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["raw_output"] += " changed"
        with self.assertRaisesRegex(CorpusError, "does not match raw_output"):
            validate_corpus(payload)

    def test_record_ids_must_be_unique(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"].append(copy.deepcopy(payload["records"][0]))
        with self.assertRaisesRegex(CorpusError, "duplicate record id"):
            validate_corpus(payload)

    def test_embedded_case_must_use_latest_schema(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["case"]["schema_version"] = 2
        with self.assertRaisesRegex(CorpusError, "case.schema_version: expected 3"):
            validate_corpus(payload)

    def test_origin_requires_a_real_timestamp_and_url(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["origin"]["generated_at"] = "2026-08-23"
        with self.assertRaisesRegex(CorpusError, "timestamp must include a UTC offset"):
            validate_corpus(payload)

        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["origin"]["source_url"] = "docs.example"
        with self.assertRaisesRegex(CorpusError, "absolute HTTP"):
            validate_corpus(payload)

    def test_pending_annotation_cannot_hide_a_label(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["annotation"]["verdict"] = "valid"
        with self.assertRaisesRegex(CorpusError, "pending annotations must not contain a label"):
            validate_corpus(payload)

    def test_completed_annotation_requires_a_rationale_and_annotator(self):
        payload = copy.deepcopy(self.corpus)
        annotation = payload["records"][0]["annotation"]
        annotation.update({"status": "labeled", "verdict": "valid"})
        with self.assertRaisesRegex(CorpusError, "completed annotations require"):
            validate_corpus(payload)

        annotation.update(
            {"rationale": "Both coupled identities simplify to zero.", "annotators": ["oo"]}
        )
        validate_corpus(payload)

    def test_failure_modes_require_an_invalid_verdict(self):
        payload = copy.deepcopy(self.corpus)
        annotation = payload["records"][0]["annotation"]
        annotation.update(
            {
                "status": "labeled",
                "verdict": "valid",
                "failure_modes": ["pde_residual"],
                "rationale": "test",
                "annotators": ["oo"],
            }
        )
        with self.assertRaisesRegex(CorpusError, "only valid for an invalid verdict"):
            validate_corpus(payload)

        annotation.update({"verdict": "invalid", "failure_modes": []})
        with self.assertRaisesRegex(CorpusError, "invalid verdicts require"):
            validate_corpus(payload)

    def test_adjudication_requires_two_annotators(self):
        payload = copy.deepcopy(self.corpus)
        annotation = payload["records"][0]["annotation"]
        annotation.update(
            {
                "status": "adjudicated",
                "verdict": "unclear",
                "rationale": "The requested solution semantics are underspecified.",
                "annotators": ["oo"],
            }
        )
        with self.assertRaisesRegex(CorpusError, "at least two annotators"):
            validate_corpus(payload)

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{")
            with self.assertRaisesRegex(CorpusError, "invalid JSON"):
                load_corpus(path)

    def test_payload_is_plain_json(self):
        json.dumps(self.corpus)


if __name__ == "__main__":
    unittest.main()
