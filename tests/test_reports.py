import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.adversarial_heat import build_cases
from pdecert import (
    BoundEvidence,
    BoundType,
    EvidenceEvent,
    EvidenceKind,
    EvidenceLevel,
    EvidenceOutcome,
    REPORT_VERSION,
    Report,
    ReportSchemaError,
    Status,
    dump_report,
    load_report,
    report_from_dict,
    verify,
)


class EvidenceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.name: case for case in build_cases()}

    def test_exact_report_round_trip_preserves_obligation_evidence(self):
        case = self.cases["exact_heat_solution"]
        report = verify(case.problem, (case.candidate,))
        payload = report.to_dict()

        self.assertEqual(payload["report_version"], REPORT_VERSION)
        self.assertEqual(payload["aggregation_policy_version"], 1)
        self.assertEqual(report_from_dict(payload).to_dict(), payload)
        discharged = [
            event for event in report.evidence_events if event.outcome is EvidenceOutcome.DISCHARGED
        ]
        self.assertTrue(discharged)
        self.assertTrue(all(event.level is EvidenceLevel.EXACT for event in discharged))

    def test_empirical_refutation_binds_the_decision_witness(self):
        case = self.cases["pde_only_boundary_trap"]
        report = verify(case.problem, (case.candidate,))

        self.assertIs(report.status, Status.REFUTED)
        counterexamples = [
            event
            for event in report.evidence_events
            if event.kind is EvidenceKind.EMPIRICAL_COUNTEREXAMPLE
        ]
        self.assertEqual(len(counterexamples), 1)
        self.assertEqual(counterexamples[0].witness, report.witness)
        self.assertEqual(report_from_dict(report.to_dict()).to_dict(), report.to_dict())

    def test_empirical_pass_remains_inconclusive(self):
        case = self.cases["below_numeric_tolerance"]
        report = verify(case.problem, (case.candidate,))

        self.assertIs(report.status, Status.INCONCLUSIVE)
        self.assertIsNone(report.decision_evidence)
        passes = [
            event for event in report.evidence_events if event.kind is EvidenceKind.EMPIRICAL_PASS
        ]
        self.assertTrue(passes)
        self.assertTrue(all(event.outcome is EvidenceOutcome.OBSERVED_PASS for event in passes))

    def test_empirical_event_cannot_discharge_an_obligation(self):
        with self.assertRaisesRegex(ValueError, "must use OBSERVED_PASS"):
            EvidenceEvent(
                obligation_id="constraint:0",
                checker="sample",
                kind=EvidenceKind.EMPIRICAL_PASS,
                outcome=EvidenceOutcome.DISCHARGED,
                level=EvidenceLevel.EMPIRICAL,
                detail="samples passed",
            )

    def test_rigorous_event_requires_bound_scope(self):
        with self.assertRaisesRegex(ValueError, "requires a bound payload"):
            EvidenceEvent(
                obligation_id="constraint:0",
                checker="interval",
                kind=EvidenceKind.RIGOROUS_BOUND,
                outcome=EvidenceOutcome.DISCHARGED,
                level=EvidenceLevel.RIGOROUS_BOUND,
                detail="claimed enclosure",
            )

    def test_bound_type_keeps_residual_distinct_from_solution_error(self):
        event = EvidenceEvent(
            obligation_id="constraint:0",
            checker="interval",
            kind=EvidenceKind.RIGOROUS_BOUND,
            outcome=EvidenceOutcome.DISCHARGED,
            level=EvidenceLevel.RIGOROUS_BOUND,
            detail="validated residual enclosure",
            bound=BoundEvidence(
                bound_type=BoundType.UNIFORM_RESIDUAL,
                quantity="absolute PDE residual",
                upper_bound=1e-8,
                norm="L_inf",
                scope="x in [0, 1]",
                assumptions=("outward-rounded interval evaluation",),
                constants={"precision_bits": 128},
            ),
        )

        self.assertEqual(event.to_dict()["bound"]["bound_type"], "UNIFORM_RESIDUAL")
        with self.assertRaises(TypeError):
            event.bound.constants["precision_bits"] = 64

    def test_unknown_report_version_is_unsupported(self):
        case = self.cases["exact_heat_solution"]
        payload = verify(case.problem, (case.candidate,)).to_dict()
        payload["report_version"] = 99

        with self.assertRaisesRegex(ReportSchemaError, "unsupported version 99"):
            report_from_dict(payload)

        payload["report_version"] = True
        with self.assertRaisesRegex(ReportSchemaError, "unsupported version True"):
            report_from_dict(payload)

    def test_summary_evidence_must_match_obligation_events(self):
        case = self.cases["exact_heat_solution"]
        payload = verify(case.problem, (case.candidate,)).to_dict()
        payload["decision_evidence"] = "RIGOROUS_BOUND"

        with self.assertRaisesRegex(ReportSchemaError, "does not match"):
            report_from_dict(payload)

    def test_non_finite_measurement_uses_strict_json(self):
        report = Report(status=Status.INCONCLUSIVE, max_sampled_residual=math.inf)
        payload = report.to_dict()

        self.assertEqual(payload["max_sampled_residual"], "infinity")
        json.dumps(payload, allow_nan=False)
        self.assertTrue(math.isinf(report_from_dict(payload).max_sampled_residual))

    def test_dump_and_load_are_deterministic(self):
        case = self.cases["exact_heat_solution"]
        report = verify(case.problem, (case.candidate,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            dump_report(report, path)
            first = path.read_bytes()
            loaded = load_report(path)
            dump_report(loaded, path)
            second = path.read_bytes()

        self.assertEqual(first, second)

    def test_loader_rejects_duplicate_keys_and_nonstandard_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"report_version": 1, "report_version": 1}')
            with self.assertRaisesRegex(ReportSchemaError, "duplicate object key"):
                load_report(duplicate)

            nonstandard = Path(directory) / "nonstandard.json"
            nonstandard.write_text('{"max_sampled_residual": NaN}')
            with self.assertRaisesRegex(ReportSchemaError, "non-standard JSON constant"):
                load_report(nonstandard)

    def test_canonical_schema_is_valid_json(self):
        payload = json.loads(Path("schema/report-v1.schema.json").read_text())
        self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(payload["properties"]["report_version"], {"const": 1})


if __name__ == "__main__":
    unittest.main()
