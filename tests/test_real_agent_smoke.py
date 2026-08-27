import json
import tempfile
import unittest
from pathlib import Path

from experiments.real_agent_smoke import build_result, heat_case, heat_prompt, write_new_json
from pdecert import SymbolicAgentSession, SymbolicAgentTool


class RealAgentSmokeTests(unittest.TestCase):
    def test_case_keeps_field_referenced_constraints(self):
        case = heat_case()
        self.assertIn("D(u, t)", case.problem.pde_residuals[0].source)
        self.assertEqual(case.field_names, ("u",))
        self.assertIn("u(x, t)", heat_prompt())

    def test_result_is_explicit_about_scope_and_keeps_raw_evidence(self):
        session = SymbolicAgentSession(
            "run-1",
            "heat-dirichlet-01",
            "test/model@revision via provider",
            SymbolicAgentTool(heat_case()),
        )
        session.submit(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"}))
        run = session.finish("The verifier returned PROVED.")
        result = build_result(
            run=run,
            model_id="test/model",
            model_revision="a" * 40,
            provider="provider",
            max_steps=4,
            max_tokens=128,
            timeout_seconds=30,
            seed=0,
            pdecert_revision="b" * 40,
            generated_at="2026-08-27T00:00:00+00:00",
        )

        self.assertIn("not independent ground-truth", result["scope"])
        self.assertEqual(result["run"]["final_output"], "The verifier returned PROVED.")
        self.assertEqual(result["summary"]["models"][0]["final_proved_runs"], 1)
        self.assertEqual(result["invocation"]["seed_requested"], 0)

    def test_result_writer_refuses_to_replace_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                write_new_json(path, {"value": 2})


if __name__ == "__main__":
    unittest.main()
