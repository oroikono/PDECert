import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.collect_atlas_batch import (
    CollectionError,
    _case_payload,
    _validate_observed_model,
    extract_fields,
    load_manifest,
    materialize,
    prompt_for,
)
from pdecert import Status, case_from_dict, load_record_bundle, verify


MANIFEST = Path("experiments/atlas_batches/qwen3_1_7b_batch01.json")


class AtlasBatchCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def test_predeclared_manifest_covers_eight_unique_cases(self):
        self.assertEqual(self.manifest["batch_version"], 1)
        self.assertEqual(len(self.manifest["cases"]), 8)
        identifiers = [case["id"] for case in self.manifest["cases"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(any(len(case["fields"]) == 2 for case in self.manifest["cases"]))

    def test_every_predeclared_problem_has_a_proved_reference_solution(self):
        references = {
            "qwen3-parametric-heat-01": {"u": "exp(-k*pi**2*t)*sin(pi*x)"},
            "qwen3-poisson-polynomial-01": {"u": "x*(1-x)+y*(1-y)"},
            "qwen3-wave-speed-two-01": {"u": "sin(pi*x)*cos(2*pi*t)"},
            "qwen3-burgers-shock-01": {"u": "1-tanh(5*(x-t))"},
            "qwen3-fisher-kpp-01": {"u": "(1+exp(x/sqrt(6)-5*t/6))**(-2)"},
            "qwen3-kdv-soliton-01": {"u": "2/cosh(x-4*t)**2"},
            "qwen3-transport-2d-01": {
                "u": "sin(pi*(x-t))*cos(pi*(y-2*t))"
            },
            "qwen3-coupled-wave-01": {
                "u": "sin(pi*x)*cos(pi*t)",
                "v": "cos(pi*x)*sin(pi*t)",
            },
        }

        for case in self.manifest["cases"]:
            loaded = case_from_dict(_case_payload(case, references[case["id"]]))
            report = verify(loaded.problem, loaded.candidate_fields, symbolic_timeout=5.0)
            self.assertEqual(report.status, Status.PROVED, case["id"])

    def test_extracts_single_and_coupled_final_fields(self):
        self.assertEqual(
            extract_fields("reasoning\nFINAL u: u(x,t) = sin(π*x)^2", ["u"]),
            {"u": "sin(pi*x)**2"},
        )
        self.assertEqual(
            extract_fields("FINAL u: sin(pi*x)\nFINAL v: cos(pi*x)", ["u", "v"]),
            {"u": "sin(pi*x)", "v": "cos(pi*x)"},
        )

    def test_rejects_missing_or_duplicate_final_fields(self):
        with self.assertRaisesRegex(CollectionError, "did not return field.*v"):
            extract_fields("FINAL u: x", ["u", "v"])
        with self.assertRaisesRegex(CollectionError, "more than once"):
            extract_fields("FINAL u: x\nFINAL u: y", ["u"])

    def test_offline_model_validation_requires_the_pinned_snapshot(self):
        revision = self.manifest["model"]["revision"]
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / revision
            snapshot.mkdir()
            download = Mock(return_value=str(snapshot))
            module = SimpleNamespace(HfApi=Mock(), snapshot_download=download)
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                _validate_observed_model(self.manifest, local_files_only=True)

        download.assert_called_once_with(
            repo_id=self.manifest["model"]["identifier"],
            revision=revision,
            local_files_only=True,
        )

    def test_offline_model_validation_rejects_a_different_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / ("0" * 40)
            snapshot.mkdir()
            module = SimpleNamespace(
                HfApi=Mock(),
                snapshot_download=Mock(return_value=str(snapshot)),
            )
            with patch.dict(sys.modules, {"huggingface_hub": module}):
                with self.assertRaisesRegex(CollectionError, "model revision mismatch"):
                    _validate_observed_model(self.manifest, local_files_only=True)

    def _transcript(self, manifest, case, response):
        return {
            "batch_id": manifest["id"],
            "case_id": case["id"],
            "generated_at": "2026-08-24T16:00:00+00:00",
            "generation": manifest["generation"],
            "model": manifest["model"],
            "prompt": prompt_for(case),
            "response": response,
            "transformers_version": "4.55.4",
        }

    def test_materializes_a_pending_bundle_without_changing_raw_output(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"] = manifest["cases"][:1]
        case = manifest["cases"][0]
        response = "analysis retained\nFINAL u: exp(-k*pi**2*t)*sin(pi*x)"
        transcript = self._transcript(manifest, case, response)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            (run_directory / f"{case['id']}.json").write_text(json.dumps(transcript))
            report = materialize(
                manifest,
                run_directory,
                root / "atlas",
                root / "report.json",
                root / "rejections",
            )
            record = load_record_bundle(root / "atlas" / "records" / case["id"])

        self.assertEqual(report["outcomes"][0]["status"], "materialized_pending_review")
        self.assertEqual(record["raw_output"], response)
        self.assertEqual(record["annotation"]["status"], "pending")
        self.assertIsNone(record["annotation"]["verdict"])
        self.assertEqual(record["origin"]["revision"], manifest["model"]["revision"])

    def test_parse_failure_is_reported_and_not_materialized(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"] = manifest["cases"][:1]
        case = manifest["cases"][0]
        transcript = self._transcript(manifest, case, "I cannot solve this problem.")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            transcript_bytes = json.dumps(transcript).encode()
            (run_directory / f"{case['id']}.json").write_bytes(transcript_bytes)
            report = materialize(
                manifest,
                run_directory,
                root / "atlas",
                root / "report.json",
                root / "rejections",
            )
            materialized = (root / "atlas" / "records" / case["id"]).exists()
            archived = root / "rejections" / f"{case['id']}.json"
            archived_bytes = archived.read_bytes()

        outcome = report["outcomes"][0]
        self.assertEqual(outcome["status"], "not_materialized")
        self.assertIn("did not return field", outcome["error"])
        self.assertEqual(archived_bytes, transcript_bytes)
        self.assertEqual(outcome["transcript"], str(archived))
        self.assertFalse(materialized)


if __name__ == "__main__":
    unittest.main()
