import builtins
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from pdecert import (
    BenchmarkError,
    CROSS_ARTIFACT_ATLAS_VERSION,
    CROSS_ARTIFACT_RECORD_VERSION,
    CorpusError,
    ReviewError,
    apply_review,
    evaluate_corpus,
    load_corpus_source,
    load_cross_artifact_atlas,
    load_cross_artifact_record_bundle,
    validate_frozen_callable_integrity,
)
from pdecert.cli import main
from experiments.review_corpus import ReviewSessionError, new_review


ATLAS = Path("corpus/matched")
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
SYMBOLIC_ID = "qwen3-fisher-kpp-01"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _update_file_digest(bundle: Path, name: str) -> None:
    record_path = bundle / "record.json"
    record = json.loads(record_path.read_text())
    referenced = bundle / record["files"][name]["path"]
    record["files"][name]["sha256"] = _sha256(referenced)
    _write_json(record_path, record)


class CrossArtifactAtlasTests(unittest.TestCase):
    def test_committed_atlas_binds_a_symbolic_and_callable_matched_pair(self):
        atlas = load_cross_artifact_atlas(ATLAS)

        self.assertEqual(atlas["atlas_version"], CROSS_ARTIFACT_ATLAS_VERSION)
        self.assertEqual(
            [record["id"] for record in atlas["records"]],
            [SYMBOLIC_ID, CALLABLE_ID],
        )
        self.assertEqual(
            {record["problem_id"] for record in atlas["records"]},
            {"fisher-kpp-classical-01"},
        )
        self.assertEqual(
            {record["artifact_type"] for record in atlas["records"]},
            {"callable_model", "symbolic_expression"},
        )
        self.assertTrue(
            all(record["annotation"]["status"] == "pending" for record in atlas["records"])
        )

    def test_generic_source_loader_dispatches_atlas_v2(self):
        atlas = load_corpus_source(ATLAS)
        self.assertEqual(atlas["atlas_version"], CROSS_ARTIFACT_ATLAS_VERSION)
        self.assertNotIn("corpus_version", atlas)

    def test_unimplemented_review_and_baseline_paths_fail_explicitly(self):
        atlas = load_cross_artifact_atlas(ATLAS)
        with self.assertRaisesRegex(ReviewSessionError, "remain pending"):
            new_review(atlas)
        with self.assertRaisesRegex(ReviewError, "not implemented"):
            apply_review(
                atlas,
                {},
                annotator="reviewer",
                confirmed_independent_review=True,
            )
        with self.assertRaisesRegex(BenchmarkError, "does not evaluate"):
            evaluate_corpus(atlas)

    def test_cli_reports_artifact_and_origin_coverage(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["corpus", "validate", str(ATLAS)])
        summary = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["atlas_version"], CROSS_ARTIFACT_ATLAS_VERSION)
        self.assertEqual(
            summary["artifact_types"],
            {"callable_model": 1, "symbolic_expression": 1},
        )
        self.assertEqual(summary["origin_kinds"], {"open_model": 1, "trained_model": 1})

    def test_core_loading_does_not_import_or_materialize_pytorch(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Atlas validation must not import PyTorch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            record = load_cross_artifact_record_bundle(ATLAS / "records" / CALLABLE_ID)

        self.assertEqual(record["artifact"]["architecture"]["dtype"], "float64")
        self.assertEqual(record["record_version"], CROSS_ARTIFACT_RECORD_VERSION)

    def test_symbolic_record_preserves_the_unedited_output_digest(self):
        record = load_cross_artifact_record_bundle(ATLAS / "records" / SYMBOLIC_ID)
        self.assertEqual(
            hashlib.sha256(record["raw_output"].encode()).hexdigest(),
            record["artifact"]["raw_output_sha256"],
        )
        self.assertEqual(
            record["artifact"]["fields"], {"u": record["raw_output"].split(":", 1)[1].strip()}
        )

    def test_callable_copy_has_a_fully_reproducible_repository_integrity_record(self):
        bundle = ATLAS / "records" / CALLABLE_ID
        integrity = validate_frozen_callable_integrity(
            bundle / "artifact.json",
            bundle / "integrity.json",
        )
        self.assertEqual(integrity["artifact_sha256"], _sha256(bundle / "artifact.json"))

    def test_tampered_artifact_bytes_are_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            artifact = copied / "records" / SYMBOLIC_ID / "artifact.json"
            artifact.write_text(artifact.read_text() + " ")
            with self.assertRaisesRegex(CorpusError, "does not match artifact.json"):
                load_cross_artifact_atlas(copied)

    def test_record_metadata_rejects_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            record_path = copied / "records" / SYMBOLIC_ID / "record.json"
            source = record_path.read_text()
            source = source.replace(
                '  "artifact_type": "symbolic_expression",',
                '  "artifact_type": "symbolic_expression",\n'
                '  "artifact_type": "symbolic_expression",',
                1,
            )
            record_path.write_text(source)
            with self.assertRaisesRegex(CorpusError, "duplicate JSON field: artifact_type"):
                load_cross_artifact_atlas(copied)

    def test_direct_record_loader_rejects_a_symlinked_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / SYMBOLIC_ID
            link.symlink_to((ATLAS / "records" / SYMBOLIC_ID).resolve(), target_is_directory=True)
            with self.assertRaisesRegex(CorpusError, "regular record directory"):
                load_cross_artifact_record_bundle(link)

    def test_atlas_manifest_cannot_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            manifest = copied / "atlas.json"
            target = copied / "manifest-target.json"
            manifest.replace(target)
            manifest.symlink_to(target.name)
            with self.assertRaisesRegex(CorpusError, "atlas.json: expected a regular file"):
                load_cross_artifact_atlas(copied)

    def test_symbolic_artifact_cannot_point_to_different_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            bundle = copied / "records" / SYMBOLIC_ID
            artifact_path = bundle / "artifact.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["raw_output_sha256"] = "0" * 64
            _write_json(artifact_path, artifact)
            _update_file_digest(bundle, "artifact")
            with self.assertRaisesRegex(CorpusError, "does not match raw-output.txt"):
                load_cross_artifact_atlas(copied)

    def test_callable_inputs_must_match_template_variable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            bundle = copied / "records" / CALLABLE_ID
            template_path = bundle / "template.json"
            template = json.loads(template_path.read_text())
            template["variables"] = ["t", "x"]
            _write_json(template_path, template)
            _update_file_digest(bundle, "problem")
            with self.assertRaisesRegex(CorpusError, "input_names.*template variables"):
                load_cross_artifact_atlas(copied)

    def test_transport_validation_rejects_a_false_integrity_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            bundle = copied / "records" / CALLABLE_ID
            integrity_path = bundle / "integrity.json"
            integrity = json.loads(integrity_path.read_text())
            integrity["artifact_sha256"] = "0" * 64
            _write_json(integrity_path, integrity)
            _update_file_digest(bundle, "integrity")
            with self.assertRaisesRegex(CorpusError, "integrity.artifact_sha256: digest mismatch"):
                load_cross_artifact_atlas(copied)

    def test_coverage_cannot_misclassify_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            coverage_path = copied / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["records"][CALLABLE_ID]["artifact_type"] = "symbolic_expression"
            _write_json(coverage_path, coverage)
            with self.assertRaisesRegex(CorpusError, "artifact_type.*does not match record"):
                load_cross_artifact_atlas(copied)

    def test_unimplemented_artifact_lanes_are_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "atlas"
            shutil.copytree(ATLAS, copied)
            record_path = copied / "records" / SYMBOLIC_ID / "record.json"
            record = json.loads(record_path.read_text())
            record["artifact_type"] = "numerical_field"
            _write_json(record_path, record)
            with self.assertRaisesRegex(CorpusError, "currently supports"):
                load_cross_artifact_atlas(copied)

    def test_public_json_schemas_are_well_formed(self):
        for path in (
            Path("schema/atlas-v2.schema.json"),
            Path("schema/atlas-v2-record-v1.schema.json"),
            Path("schema/symbolic-artifact-v1.schema.json"),
        ):
            self.assertIsInstance(json.loads(path.read_text()), dict)


if __name__ == "__main__":
    unittest.main()
