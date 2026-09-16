import json
import math
import tempfile
import unittest
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

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
    Witness,
    dump_report,
    load_report,
    report_from_dict,
    verify,
)


@pytest.fixture
def decision_events():
    """Synthetic records that exercise report consistency, not PDE correctness."""

    witness = Witness("synthetic residual", {"x": 0.5}, 1.0, "synthetic nonzero residual")
    common = {"obligation_id": "constraint:0", "checker": "synthetic-checker"}
    return {
        "discharged": EvidenceEvent(
            **common,
            kind=EvidenceKind.EXACT_CERTIFICATE,
            outcome=EvidenceOutcome.DISCHARGED,
            level=EvidenceLevel.EXACT,
            detail="synthetic exact discharge",
        ),
        "bound": EvidenceEvent(
            **common,
            kind=EvidenceKind.RIGOROUS_BOUND,
            outcome=EvidenceOutcome.DISCHARGED,
            level=EvidenceLevel.RIGOROUS_BOUND,
            detail="synthetic bound discharge",
            bound=BoundEvidence(
                bound_type=BoundType.UNIFORM_RESIDUAL,
                quantity="synthetic residual",
                upper_bound=0.0,
                norm="L_inf",
                scope="synthetic test domain",
            ),
        ),
        "abstained": EvidenceEvent(
            **common,
            kind=EvidenceKind.ABSTENTION,
            outcome=EvidenceOutcome.ABSTAINED,
            level=None,
            detail="synthetic earlier abstention",
        ),
        "observed_pass": EvidenceEvent(
            **common,
            kind=EvidenceKind.EMPIRICAL_PASS,
            outcome=EvidenceOutcome.OBSERVED_PASS,
            level=EvidenceLevel.EMPIRICAL,
            detail="synthetic sampled pass",
        ),
        "exact_refutation": EvidenceEvent(
            **common,
            kind=EvidenceKind.EXACT_CERTIFICATE,
            outcome=EvidenceOutcome.REFUTED,
            level=EvidenceLevel.EXACT,
            detail="synthetic exact refutation",
            witness=witness,
        ),
        "empirical_refutation": EvidenceEvent(
            **common,
            kind=EvidenceKind.EMPIRICAL_COUNTEREXAMPLE,
            outcome=EvidenceOutcome.REFUTED,
            level=EvidenceLevel.EMPIRICAL,
            detail="synthetic empirical refutation",
            witness=witness,
        ),
    }


@pytest.fixture(scope="module")
def report_validator():
    schema = json.loads(Path("schema/report-v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("status", [Status.PROVED, Status.INCONCLUSIVE])
@pytest.mark.parametrize("refutation", ["exact_refutation", "empirical_refutation"])
@pytest.mark.parametrize("refutation_first", [False, True])
@pytest.mark.parametrize("reader", ["dictionary", "file", "schema"])
def test_non_refuted_report_rejects_refuting_evidence(
    status, refutation, refutation_first, reader, decision_events, report_validator, tmp_path
):
    events = [decision_events["discharged"], decision_events[refutation]]
    if refutation_first:
        events.reverse()
    payload = Report(
        status=status,
        decision_evidence=EvidenceLevel.EXACT if status is Status.PROVED else None,
        evidence_events=events,
    ).to_dict()

    if reader == "schema":
        with pytest.raises(ValidationError):
            report_validator.validate(payload)
    else:
        path = tmp_path / "contradictory-report.json"
        if reader == "file":
            path.write_text(json.dumps(payload, allow_nan=False))
        with pytest.raises(
            ReportSchemaError,
            match=r"^\$\.evidence_events: refuting evidence requires a REFUTED report$",
        ):
            if reader == "file":
                load_report(path)
            else:
                report_from_dict(payload)


@pytest.mark.parametrize(
    ("status", "level", "history"),
    [
        (
            Status.REFUTED,
            EvidenceLevel.EXACT,
            ("abstained", "discharged", "exact_refutation"),
        ),
        (
            Status.REFUTED,
            EvidenceLevel.EMPIRICAL,
            ("abstained", "discharged", "empirical_refutation"),
        ),
        (Status.PROVED, EvidenceLevel.EXACT, ("abstained", "observed_pass", "discharged")),
        (Status.INCONCLUSIVE, None, ("discharged", "abstained", "observed_pass")),
        (Status.INCONCLUSIVE, None, ("discharged",)),
        (Status.INCONCLUSIVE, None, ("abstained",)),
        (Status.INCONCLUSIVE, None, ("observed_pass",)),
        (Status.INCONCLUSIVE, None, ()),
        (Status.PROVED, EvidenceLevel.EXACT, ("bound", "discharged")),
        (Status.PROVED, EvidenceLevel.EXACT, ("discharged", "bound")),
        (Status.PROVED, EvidenceLevel.RIGOROUS_BOUND, ("bound",)),
    ],
)
def test_report_accepts_compatible_evidence_history(
    status, level, history, decision_events, report_validator, tmp_path
):
    events = [decision_events[name] for name in history]
    payload = Report(
        status=status,
        decision_evidence=level,
        witness=next((event.witness for event in events if event.witness is not None), None),
        incomplete_reasons=(
            {"synthetic residual": "synthetic earlier abstention"} if "abstained" in history else {}
        ),
        evidence_events=events,
    ).to_dict()

    report_validator.validate(payload)
    assert report_from_dict(payload).to_dict() == payload
    path = tmp_path / "compatible-report.json"
    path.write_text(json.dumps(payload, allow_nan=False))
    assert load_report(path).to_dict() == payload


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
