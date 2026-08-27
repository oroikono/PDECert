import json
import unittest
from unittest.mock import patch

from pdecert import SymbolicAgentSession, SymbolicAgentTool, case_from_dict
from pdecert.integrations.smolagents import (
    build_smolagents_tool,
    run_smolagents_symbolic_agent,
)


try:
    from smolagents import Tool
except ImportError:
    Tool = None


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


@unittest.skipIf(Tool is None, "smolagents optional dependency is not installed")
class SmolagentsIntegrationTests(unittest.TestCase):
    def test_adapter_is_a_real_smolagents_tool(self):
        session = SymbolicAgentSession(
            "run-1",
            "heat-01",
            "test/model",
            SymbolicAgentTool(heat_case()),
        )
        tool = build_smolagents_tool(session)
        result = json.loads(tool(candidate_fields_json=json.dumps({"u": "0"})))
        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "pdecert_verify_symbolic_candidate")
        self.assertEqual(result["report"]["status"], "REFUTED")

    def test_runner_constructs_tool_calling_agent_and_records_its_calls(self):
        class FakeToolCallingAgent:
            def __init__(self, *, tools, model, max_steps, **options):
                self.tool = tools[0]
                self.model = model
                self.max_steps = max_steps
                self.options = options

            def run(self, task):
                self.tool(candidate_fields_json=json.dumps({"u": "0"}))
                self.tool(candidate_fields_json=json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"}))
                return f"completed: {task[-7:]}"

        with patch("smolagents.ToolCallingAgent", FakeToolCallingAgent):
            run = run_smolagents_symbolic_agent(
                trusted_case=heat_case(),
                model=object(),
                prompt="Solve and verify the heat equation.",
                run_id="run-1",
                problem_id="heat-01",
                generator="test/model",
                max_steps=4,
            )

        self.assertEqual(len(run.evaluations), 2)
        self.assertEqual(run.evaluations[0].report.status.value, "REFUTED")
        self.assertEqual(run.evaluations[1].report.status.value, "PROVED")


if __name__ == "__main__":
    unittest.main()
