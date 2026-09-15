"""Run the published walkthrough verbatim, including in core-only package CI."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "check-your-candidate.md"


class CandidateWalkthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guide = GUIDE.read_text(encoding="utf-8")
        blocks = re.findall(
            r"<!-- tested-candidate-walkthrough:start -->\n(.*?)"
            r"<!-- tested-candidate-walkthrough:end -->",
            guide,
            re.DOTALL,
        )
        if len(blocks) != 1:
            raise AssertionError("expected exactly one tested walkthrough block")
        recipes = re.findall(r"^python - (.*?) <<'PY'\n(.*?)^PY$", blocks[0], re.M | re.S)
        if len(recipes) != 1:
            raise AssertionError("expected exactly one Python heredoc")
        arguments = shlex.split(recipes[0][0])
        if len(arguments) != 1:
            raise AssertionError("expected exactly one candidate argument")
        cls.default_candidate = arguments[0]
        cls.source = recipes[0][1]

    def run_candidate(self, expression=None):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable]
        if sys.flags.isolated:
            command.append("-I")
            env.pop("PYTHONPATH", None)
        # Preserve source-test imports when the child runs outside the checkout.
        # Clean wheel/sdist jobs do not supply PYTHONPATH or install test extras.
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = os.pathsep.join(
                str(Path(entry).resolve()) for entry in env["PYTHONPATH"].split(os.pathsep)
            )
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                command + ["-c", self.source, expression or self.default_candidate],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def successful_payload(self, expression=None):
        result = self.run_candidate(expression)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def test_verbatim_default_proves_the_declared_problem(self):
        payload = self.successful_payload()
        self.assertEqual(payload["report"]["status"], "PROVED")
        self.assertEqual(payload["report"]["decision_evidence"], "EXACT")
        self.assertIsNone(payload["report"]["witness"])
        self.assertEqual(payload["case"]["schema_version"], 3)
        self.assertEqual(payload["case"]["variables"], ["x", "t"])
        self.assertEqual(len(payload["case"]["conditions"]), 3)
        self.assertEqual(payload["settings"]["tolerance"], 1e-9)
        self.assertEqual(payload["settings"]["samples_per_axis"], 5)

    def test_changed_candidate_keeps_the_pde_but_fails_the_initial_data(self):
        payload = self.successful_payload(self.default_candidate + " + x/10")
        report = payload["report"]
        self.assertEqual(report["status"], "REFUTED")
        self.assertEqual(report["decision_evidence"], "EMPIRICAL")
        self.assertEqual(report["exact_checks"]["heat PDE"], "identity")
        self.assertEqual(report["witness"]["constraint"], "initial condition")
        self.assertEqual(report["witness"]["point"], {"x": 0.113})
        self.assertAlmostEqual(report["witness"]["residual"], 0.0113)

    def test_below_tolerance_error_does_not_become_a_proof(self):
        payload = self.successful_payload(self.default_candidate + " + t*x*(1-x)/10**14")
        report = payload["report"]
        self.assertEqual(report["status"], "INCONCLUSIVE")
        self.assertIsNone(report["decision_evidence"])
        self.assertIsNone(report["witness"])
        self.assertGreater(report["max_sampled_residual"], 0)
        self.assertLess(report["max_sampled_residual"], payload["settings"]["tolerance"])
        self.assertTrue(report["incomplete_reasons"])
        self.assertTrue(
            any(event["kind"] == "EMPIRICAL_PASS" for event in report["evidence_events"])
        )

    def test_unsupported_candidate_syntax_is_an_input_error_not_a_verdict(self):
        for expression in ("sin(y)", "__import__('os')"):
            with self.subTest(expression=expression):
                result = self.run_candidate(expression)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("TemplateError", result.stderr)


if __name__ == "__main__":
    unittest.main()
