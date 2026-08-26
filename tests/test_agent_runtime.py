import json
import unittest

from pdecert import (
    AgentRun,
    Status,
    SymbolicAgentSession,
    SymbolicAgentTool,
    case_from_dict,
    summarize_agent_runs,
)


def heat_case():
    return case_from_dict(
        {
            "schema_version": 3,
            "name": "heat equation",
            "variables": ["x", "t"],
            "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            "parameters": {},
            "pde_residuals": [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}],
            "conditions": [
                {"name": "initial condition", "expression": "At(u, t, 0) - sin(pi*x)"},
                {"name": "left boundary", "expression": "At(u, x, 0)"},
                {"name": "right boundary", "expression": "At(u, x, 1)"},
            ],
            "fields": {"u": "exp(-pi**2*t)*sin(pi*x)"},
        }
    )


class AgentRuntimeTests(unittest.TestCase):
    def make_session(self, generator="test/model-a", run_id="run-1"):
        return SymbolicAgentSession(
            run_id=run_id,
            problem_id="heat-01",
            generator=generator,
            verifier=SymbolicAgentTool(heat_case()),
            metadata={"seed": "7"},
        )

    def test_session_records_rejected_proposal_counterexample_and_repair(self):
        session = self.make_session()
        malformed = json.loads(session.submit("not-json"))
        wrong = json.loads(session.submit(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x) + t"})))
        exact = json.loads(session.submit(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"})))
        run = session.finish("The verifier returned PROVED.")

        self.assertFalse(malformed["ok"])
        self.assertEqual(wrong["report"]["status"], Status.REFUTED.value)
        self.assertEqual(exact["report"]["status"], Status.PROVED.value)
        self.assertEqual(len(run.tool_calls), 3)
        self.assertEqual(len(run.evaluations), 2)
        self.assertEqual(
            run.evaluations[1].proposal.parent_proposal_id,
            run.evaluations[0].proposal.proposal_id,
        )

    def test_run_serialization_hides_raw_model_text_by_default(self):
        session = self.make_session()
        candidate = json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"})
        session.submit(candidate)
        run = session.finish("private final response")

        public = run.to_dict()
        self.assertNotIn("final_output", public)
        self.assertNotIn("candidate_fields_json", public["tool_calls"][0])
        self.assertNotIn("raw_output", public["trace"]["evaluations"][0]["proposal"])

        private = run.to_dict(include_raw_outputs=True)
        self.assertEqual(private["final_output"], "private final response")
        self.assertEqual(private["tool_calls"][0]["candidate_fields_json"], candidate)

    def test_session_cannot_be_used_after_finish(self):
        session = self.make_session()
        session.finish("done")
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            session.submit(json.dumps({"u": "0"}))
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            session.finish("again")

    def test_metrics_compare_exact_generator_identities_without_accuracy_claims(self):
        repaired_session = self.make_session()
        repaired_session.submit(json.dumps({"u": "0"}))
        repaired_session.submit(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"}))
        repaired = repaired_session.finish("done")

        empty = AgentRun("run-2", "heat-01", "test/model-b", (), "no tool call")
        report = summarize_agent_runs((repaired, empty)).to_dict()
        by_model = {item["generator"]: item for item in report["models"]}

        self.assertIn("not independent ground-truth accuracy", report["metric_scope"])
        self.assertEqual(by_model["test/model-a"]["repaired_to_proved_runs"], 1)
        self.assertEqual(by_model["test/model-a"]["final_proved_rate"], 1.0)
        self.assertEqual(
            by_model["test/model-b"]["final_status_counts"],
            {"NO_VALID_PROPOSAL": 1},
        )

    def test_metrics_reject_non_run_items(self):
        with self.assertRaisesRegex(TypeError, "AgentRun"):
            summarize_agent_runs([object()])


if __name__ == "__main__":
    unittest.main()
