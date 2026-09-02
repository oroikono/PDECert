import json
import tempfile
import unittest
from pathlib import Path

from pdecert import (
    EvidenceLevel,
    FrozenCallableError,
    Status,
    load_case,
    load_template,
    template_from_case,
    template_to_dict,
    validate_frozen_callable_integrity,
    verify,
)

from experiments.trained_fisher_kpp_pair import (
    DEFAULT_CASE,
    DEFAULT_FIXTURE,
    DEFAULT_INTEGRITY,
    DEFAULT_RAW,
    DEFAULT_RECORD,
    DEFAULT_TEMPLATE,
    build_case,
    build_symbolic_case,
    load_symbolic_proposal,
    run,
)


RESULT = Path("results/trained-fisher-kpp-pair.json")
EXPECTED_RAW_SHA256 = "fd4496138199cf29b9b0d3a829fd7ffd7a986711664b19ac01788e3911653ef1"


class FisherKppProvenanceTests(unittest.TestCase):
    def test_candidate_free_template_matches_the_preserved_corpus_problem(self):
        template = load_template(DEFAULT_TEMPLATE)
        corpus_template = template_from_case(load_case(DEFAULT_CASE))
        self.assertEqual(template_to_dict(template), template_to_dict(corpus_template))

    def test_raw_symbolic_proposal_keeps_digest_and_pending_review_state(self):
        expression, provenance = load_symbolic_proposal()
        self.assertEqual(expression, "(1 + exp(x/sqrt(6) - 5*t/6))**(-2)")
        self.assertEqual(
            provenance["files"]["raw_output"]["sha256"],
            EXPECTED_RAW_SHA256,
        )
        self.assertEqual(provenance["annotation"]["status"], "pending")
        self.assertEqual(provenance["annotation"]["annotators"], [])

    def test_raw_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw-output.txt"
            case = root / "case.json"
            record = root / "record.json"
            raw.write_text(DEFAULT_RAW.read_text() + "\n")
            case.write_bytes(DEFAULT_CASE.read_bytes())
            record.write_bytes(DEFAULT_RECORD.read_bytes())
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_symbolic_proposal(raw, case, record)

    def test_integrity_claim_rejects_an_unbound_active_input_path(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            copied_raw = Path(directory) / "raw-output.txt"
            copied_raw.write_bytes(DEFAULT_RAW.read_bytes())
            with self.assertRaisesRegex(FrozenCallableError, "not bound"):
                run(raw=copied_raw)

    def test_preserved_symbolic_candidate_has_exact_evidence(self):
        _, symbolic_case, _ = build_symbolic_case()
        report = verify(symbolic_case.problem, symbolic_case.candidate_fields)
        self.assertEqual(report.status, Status.PROVED)
        self.assertEqual(report.decision_evidence, EvidenceLevel.EXACT)


try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "trained fixture requires PyTorch")
class TrainedFisherKppPairTests(unittest.TestCase):
    def test_frozen_artifact_and_bound_sources_have_valid_digests(self):
        self.assertTrue(DEFAULT_FIXTURE.is_file(), "the Fisher--KPP fixture is missing")
        self.assertTrue(DEFAULT_INTEGRITY.is_file(), "the Fisher--KPP integrity record is missing")
        validate_frozen_callable_integrity(DEFAULT_FIXTURE, DEFAULT_INTEGRITY)

    def test_matched_lanes_keep_exact_and_empirical_evidence_separate(self):
        from pdecert import LaneVerificationOptions, verify_matched_case

        matched, _, _ = build_case()
        report = verify_matched_case(
            matched,
            options={
                "symbolic-qwen3": LaneVerificationOptions(symbolic_timeout=2.0),
                "trained-pinn": LaneVerificationOptions(
                    tolerance=1e-3,
                    samples_per_axis=6,
                ),
            },
        )
        self.assertEqual(report.reports["symbolic-qwen3"].status, Status.PROVED)
        self.assertEqual(
            report.reports["symbolic-qwen3"].decision_evidence,
            EvidenceLevel.EXACT,
        )
        self.assertEqual(report.reports["trained-pinn"].status, Status.REFUTED)
        self.assertEqual(
            report.reports["trained-pinn"].decision_evidence,
            EvidenceLevel.EMPIRICAL,
        )
        witness = report.reports["trained-pinn"].witness
        self.assertIsNotNone(witness)
        self.assertGreater(witness.residual, 1e-3)
        self.assertNotIn("status", report.to_dict())

        committed = json.loads(RESULT.read_text())["matched_report"]
        committed_lanes = {lane["name"]: lane["report"] for lane in committed["lanes"]}
        for name, live in report.reports.items():
            self.assertEqual(committed_lanes[name]["status"], live.status.value)
            self.assertEqual(
                committed_lanes[name]["decision_evidence"],
                live.decision_evidence.value,
            )

    def test_committed_result_names_the_same_artifact_and_raw_proposal(self):
        result = json.loads(RESULT.read_text())
        integrity = json.loads(DEFAULT_INTEGRITY.read_text())
        self.assertEqual(result["fixture"]["sha256"], integrity["artifact_sha256"])
        self.assertEqual(
            result["symbolic_proposal"]["files"]["raw_output"]["sha256"],
            EXPECTED_RAW_SHA256,
        )
        self.assertEqual(result["symbolic_proposal"]["annotation"]["status"], "pending")
        self.assertNotIn("status", result["matched_report"])
        self.assertEqual(
            result["matched_report"]["lanes"][1]["report"]["status"],
            "REFUTED",
        )
        self.assertEqual(
            result["matched_report"]["lanes"][1]["report"]["decision_evidence"],
            "EMPIRICAL",
        )
        self.assertGreater(
            result["matched_report"]["lanes"][1]["report"]["witness"]["residual"],
            result["evaluation"]["callable_tolerance"],
        )
        self.assertEqual(
            result["runtime"]["torch_version"], result["fixture"]["training"]["torch_version"]
        )

    def test_integrity_binds_decision_relevant_evaluator_sources(self):
        integrity = json.loads(DEFAULT_INTEGRITY.read_text())
        bound = set(integrity["source_files_sha256"])
        self.assertTrue(
            {
                "src/pdecert/artifacts.py",
                "src/pdecert/autodiff.py",
                "src/pdecert/checks.py",
                "src/pdecert/compiler.py",
                "src/pdecert/core.py",
                "src/pdecert/evidence.py",
                "src/pdecert/matched.py",
                "src/pdecert/schema.py",
                "src/pdecert/templates.py",
            }.issubset(bound)
        )


if __name__ == "__main__":
    unittest.main()
