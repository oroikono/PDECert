import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.coupled_wave import build_case
from pdecert import (
    ATLAS_VERSION,
    CorpusError,
    case_to_dict,
    dump_atlas,
    dump_corpus,
    load_atlas,
    load_corpus,
    load_corpus_source,
    load_record_bundle,
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

    def _write_bundle(self, root: Path, record: dict, directory_name: str | None = None) -> Path:
        bundle = root / "records" / (directory_name or record["id"])
        bundle.mkdir(parents=True)
        metadata = {
            "annotation": record["annotation"],
            "id": record["id"],
            "origin": record["origin"],
            "output_sha256": record["output_sha256"],
        }
        (bundle / "record.json").write_text(json.dumps(metadata))
        (bundle / "case.json").write_text(json.dumps(record["case"]))
        (bundle / "raw-output.txt").write_text(record["raw_output"])
        return bundle

    def _write_manifest(self, root: Path) -> None:
        manifest = {
            "atlas_version": ATLAS_VERSION,
            "name": "test atlas",
            "description": "A modular atlas used by the unit tests.",
        }
        (root / "atlas.json").write_text(json.dumps(manifest))

    def test_valid_record_passes(self):
        validate_corpus(self.corpus)

    def test_file_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            dump_corpus(self.corpus, path)
            loaded = load_corpus(path)
            self.assertTrue(path.read_text().endswith("\n"))
        self.assertEqual(loaded, self.corpus)

    def test_modular_record_reconstructs_the_corpus_record(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._write_bundle(Path(directory), self.record)
            loaded = load_record_bundle(bundle)

        self.assertEqual(loaded, self.record)

    def test_modular_atlas_loads_records_in_directory_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root)
            second = copy.deepcopy(self.record)
            second["id"] = "sympy-wave-002"
            self._write_bundle(root, second)
            self._write_bundle(root, self.record)
            loaded = load_atlas(root)

        self.assertEqual(loaded["name"], "test atlas")
        self.assertEqual(
            [record["id"] for record in loaded["records"]],
            ["sympy-wave-001", "sympy-wave-002"],
        )

    def test_modular_atlas_dump_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atlas"
            dump_atlas(self.corpus, output)
            loaded = load_atlas(output)

            self.assertTrue((output / "atlas.json").read_text().endswith("\n"))
            self.assertTrue(
                (output / "records" / self.record["id"] / "record.json")
                .read_text()
                .endswith("\n")
            )
        self.assertEqual(loaded, self.corpus)

    def test_modular_atlas_dump_preserves_raw_utf8_bytes(self):
        corpus = copy.deepcopy(self.corpus)
        raw_output = "u = α + β\r\n"
        corpus["records"][0]["raw_output"] = raw_output
        corpus["records"][0]["output_sha256"] = output_sha256(raw_output)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atlas"
            dump_atlas(corpus, output)
            stored = (
                output / "records" / self.record["id"] / "raw-output.txt"
            ).read_bytes()

        self.assertEqual(stored, raw_output.encode())

    def test_modular_atlas_dump_refuses_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atlas"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve me")
            with self.assertRaisesRegex(CorpusError, "refusing to overwrite"):
                dump_atlas(self.corpus, output)

            self.assertEqual(marker.read_text(), "preserve me")

    def test_modular_record_directory_must_match_its_id(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._write_bundle(Path(directory), self.record, "wrong-name")
            with self.assertRaisesRegex(CorpusError, "directory name must match"):
                load_record_bundle(bundle)

    def test_modular_record_preserves_crlf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            record = copy.deepcopy(self.record)
            record["raw_output"] += "\r\n"
            record["output_sha256"] = output_sha256(record["raw_output"])
            bundle = self._write_bundle(Path(directory), record)
            loaded = load_record_bundle(bundle)

        self.assertEqual(loaded["raw_output"], record["raw_output"])

    def test_modular_record_rejects_unexpected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._write_bundle(Path(directory), self.record)
            (bundle / "notes.txt").write_text("not part of the record")
            with self.assertRaisesRegex(CorpusError, "unexpected bundle file"):
                load_record_bundle(bundle)

    def test_modular_record_preserves_raw_output_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._write_bundle(Path(directory), self.record)
            (bundle / "raw-output.txt").write_text("modified")
            with self.assertRaisesRegex(CorpusError, "does not match raw_output"):
                load_record_bundle(bundle)

    def test_modular_atlas_rejects_loose_files_in_records_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root)
            records = root / "records"
            records.mkdir()
            (records / "notes.txt").write_text("not a record bundle")
            with self.assertRaisesRegex(CorpusError, "only record directories"):
                load_atlas(root)

    def test_modular_atlas_rejects_an_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root)
            manifest = json.loads((root / "atlas.json").read_text())
            manifest["atlas_version"] = ATLAS_VERSION + 1
            (root / "atlas.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(CorpusError, "atlas_version: expected 1"):
                load_atlas(root)

    def test_corpus_source_accepts_an_empty_modular_atlas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root)
            loaded = load_corpus_source(root)

        self.assertEqual(loaded["records"], [])

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

    def test_synthetic_origin_is_explicitly_supported(self):
        payload = copy.deepcopy(self.corpus)
        payload["records"][0]["origin"]["kind"] = "synthetic"
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
