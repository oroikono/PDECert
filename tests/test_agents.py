import json
import unittest

import sympy as sp

from experiments.adversarial_heat import build_cases
from pdecert import (
    AgentEvaluation,
    AgentProposal,
    AgentTrace,
    Status,
    SymbolicAgentTool,
    SymbolicCandidate,
    VerificationCase,
    case_from_dict,
    case_to_dict,
    evaluate_agent_proposal,
)


def proposal(proposal_id, candidate, *, parent=None, raw_output="candidate"):
    return AgentProposal(
        proposal_id=proposal_id,
        generator="test-agent/model-v1",
        artifact=SymbolicCandidate.from_expressions({"u": candidate}),
        raw_output=raw_output,
        parent_proposal_id=parent,
        metadata={"seed": "7"},
    )


class AgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exact_case = build_cases()[0]
        seed_case = VerificationCase(
            cls.exact_case.problem,
            (cls.exact_case.candidate,),
            ("u",),
        )
        payload = case_to_dict(seed_case)
        payload["pde_residuals"] = [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}]
        payload["conditions"] = [
            {
                "name": "initial condition",
                "expression": "At(u, t, 0) - sin(pi*x)",
            },
            {"name": "left boundary", "expression": "At(u, x, 0)"},
            {"name": "right boundary", "expression": "At(u, x, 1)"},
        ]
        cls.trusted_case = case_from_dict(payload)

    def test_proposal_preserves_raw_output_provenance(self):
        item = proposal("attempt-1", self.exact_case.candidate, raw_output="u = exact")
        payload = item.to_dict()
        self.assertNotIn("raw_output", payload)
        self.assertEqual(len(payload["raw_output_sha256"]), 64)
        self.assertEqual(item.to_dict(include_raw_output=True)["raw_output"], "u = exact")
        with self.assertRaises(TypeError):
            item.metadata["seed"] = "8"

    def test_exact_and_wrong_proposals_use_normal_verifier_statuses(self):
        exact = evaluate_agent_proposal(
            self.trusted_case,
            proposal("attempt-1", self.exact_case.candidate),
        )
        wrong_candidate = self.exact_case.candidate + sp.Symbol("t", real=True)
        wrong = evaluate_agent_proposal(
            self.trusted_case,
            proposal("attempt-2", wrong_candidate),
        )
        self.assertIs(exact.report.status, Status.PROVED)
        self.assertIs(wrong.report.status, Status.REFUTED)
        self.assertIsNotNone(wrong.report.witness)

    def test_trace_records_a_repair_chain_without_making_it_ground_truth(self):
        wrong = AgentEvaluation(
            proposal("attempt-1", self.exact_case.candidate + 1, raw_output="u + 1"),
            evaluate_agent_proposal(
                self.trusted_case,
                proposal("temporary", self.exact_case.candidate + 1),
            ).report,
        )
        repaired = evaluate_agent_proposal(
            self.trusted_case,
            proposal(
                "attempt-2",
                self.exact_case.candidate,
                parent="attempt-1",
                raw_output="u",
            ),
        )
        trace = AgentTrace("run-1", "heat-01", (wrong, repaired))
        payload = trace.to_dict()
        self.assertNotIn("raw_output", payload["evaluations"][0]["proposal"])
        self.assertEqual(
            payload["evaluations"][1]["proposal"]["parent_proposal_id"],
            "attempt-1",
        )
        self.assertEqual(
            trace.to_dict(include_raw_outputs=True)["evaluations"][1]["proposal"]["raw_output"],
            "u",
        )

    def test_trace_rejects_missing_or_forward_parent(self):
        evaluation = evaluate_agent_proposal(
            self.trusted_case,
            proposal("attempt-2", self.exact_case.candidate, parent="attempt-1"),
        )
        with self.assertRaisesRegex(ValueError, "does not precede"):
            AgentTrace("run-1", "heat-01", (evaluation,))

    def test_symbolic_tool_keeps_problem_outside_agent_payload(self):
        tool = SymbolicAgentTool(self.trusted_case)
        exact = tool.evaluate(json.dumps({"u": sp.sstr(self.exact_case.candidate)}))
        self.assertTrue(exact["ok"])
        self.assertEqual(exact["report"]["status"], Status.PROVED.value)

        untrusted_problem = tool.evaluate(json.dumps({"pde_residuals": []}))
        self.assertFalse(untrusted_problem["ok"])
        self.assertIn("fields must be exactly", untrusted_problem["error"])

    def test_symbolic_agent_rejects_a_case_without_field_referenced_sources(self):
        unbound = VerificationCase(
            self.exact_case.problem,
            (self.exact_case.candidate,),
            ("u",),
        )
        with self.assertRaisesRegex(ValueError, "field-referenced"):
            SymbolicAgentTool(unbound)

    def test_symbolic_tool_returns_counterexamples_and_safe_input_errors(self):
        tool = SymbolicAgentTool(self.trusted_case)
        wrong = tool.evaluate(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x) + t"}))
        self.assertTrue(wrong["ok"])
        self.assertEqual(wrong["report"]["status"], Status.REFUTED.value)
        self.assertIsNotNone(wrong["report"]["witness"])

        malformed = tool.evaluate("not-json")
        self.assertFalse(malformed["ok"])
        malicious = tool.evaluate(json.dumps({"u": "__import__('os').system('id')"}))
        self.assertFalse(malicious["ok"])
        self.assertIn("unsupported expression syntax", malicious["error"])

    def test_symbolic_tool_has_a_bounded_deterministic_json_interface(self):
        tool = SymbolicAgentTool(self.trusted_case, max_payload_bytes=10)
        oversized = tool.evaluate(json.dumps({"u": "x" * 20}))
        self.assertFalse(oversized["ok"])
        rendered = SymbolicAgentTool(self.trusted_case)(json.dumps({"u": "0"}))
        self.assertEqual(rendered, json.dumps(json.loads(rendered), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
