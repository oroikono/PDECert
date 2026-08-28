import io
import json
import unittest
from contextlib import redirect_stdout

from pdecert import QUICKSTART_VERSION, Status, report_from_dict, run_quickstart
from pdecert.cli import main


class QuickstartTests(unittest.TestCase):
    def test_quickstart_reproduces_all_three_decisions(self):
        payload = run_quickstart()

        self.assertEqual(payload["quickstart_version"], QUICKSTART_VERSION)
        self.assertTrue(payload["passed"])
        self.assertEqual(
            [scenario["observed_status"] for scenario in payload["scenarios"]],
            [Status.PROVED.value, Status.REFUTED.value, Status.INCONCLUSIVE.value],
        )
        self.assertTrue(all(payload["checks"].values()))

    def test_quickstart_reports_round_trip_through_public_schema(self):
        payload = run_quickstart()

        for scenario in payload["scenarios"]:
            restored = report_from_dict(scenario["report"])
            self.assertEqual(restored.status.value, scenario["observed_status"])

    def test_sampled_pass_never_becomes_proof(self):
        payload = run_quickstart()
        scenario = next(
            item for item in payload["scenarios"] if item["id"] == "sampled-pass-abstention"
        )

        self.assertEqual(scenario["report"]["status"], Status.INCONCLUSIVE.value)
        self.assertIsNone(scenario["report"]["decision_evidence"])
        self.assertTrue(
            any(
                event["kind"] == "EMPIRICAL_PASS" for event in scenario["report"]["evidence_events"]
            )
        )

    def test_agent_trace_links_rejected_proposal_to_exact_repair(self):
        trace = run_quickstart()["agent_trace"]
        rejected, repaired = trace["evaluations"]

        self.assertEqual(rejected["verification"]["status"], Status.REFUTED.value)
        self.assertIsNotNone(rejected["verification"]["witness"])
        self.assertEqual(repaired["verification"]["status"], Status.PROVED.value)
        self.assertEqual(
            repaired["proposal"]["parent_proposal_id"],
            rejected["proposal"]["proposal_id"],
        )
        self.assertNotIn("raw_output", rejected["proposal"])
        self.assertEqual(len(rejected["proposal"]["raw_output_sha256"]), 64)

    def test_quickstart_payload_is_deterministic_and_strict_json(self):
        first = run_quickstart()
        second = run_quickstart()

        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_cli_json_output_is_machine_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["quickstart", "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["quickstart_version"], QUICKSTART_VERSION)

    def test_cli_human_output_explains_the_evidence_boundary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["quickstart"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("PROVED (EXACT)", rendered)
        self.assertIn("REFUTED (EMPIRICAL)", rendered)
        self.assertIn("INCONCLUSIVE (no decisive evidence)", rendered)
        self.assertIn("REFUTED -> PROVED", rendered)
        self.assertIn("sampling may refute but never proves", rendered)


if __name__ == "__main__":
    unittest.main()
