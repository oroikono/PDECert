import unittest

import sympy as sp

from experiments.adversarial_heat import build_cases
from pdecert import (
    CallableCandidate,
    Status,
    SymbolicCandidate,
    verify,
    verify_artifact,
)


class SolutionArtifactTests(unittest.TestCase):
    def test_symbolic_candidate_preserves_named_field_order(self):
        x = sp.symbols("x", real=True)
        artifact = SymbolicCandidate.from_expressions({"velocity": x, "pressure": x**2})
        self.assertEqual(artifact.kind, "symbolic")
        self.assertEqual(artifact.field_names, ("velocity", "pressure"))

    def test_symbolic_candidate_rejects_duplicate_names(self):
        x = sp.symbols("x", real=True)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            SymbolicCandidate((("u", x), ("u", x**2)))

    def test_callable_candidate_validates_backend_and_fields(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            CallableCandidate(())
        with self.assertRaisesRegex(ValueError, "only supported.*torch"):
            CallableCandidate.from_mapping({"u": lambda points: points}, backend="jax")
        with self.assertRaisesRegex(TypeError, "must be callable"):
            CallableCandidate((("u", object()),))

    def test_legacy_verify_and_symbolic_artifact_have_identical_reports(self):
        case = build_cases()[0]
        artifact = SymbolicCandidate.from_expressions((case.candidate,))
        legacy = verify(case.problem, (case.candidate,)).to_dict()
        typed = verify_artifact(case.problem, artifact).to_dict()
        self.assertEqual(legacy, typed)
        self.assertEqual(typed["status"], Status.PROVED.value)

    def test_mismatched_problem_and_artifact_are_rejected(self):
        case = build_cases()[0]
        artifact = CallableCandidate.from_mapping({"u": lambda points: points})
        with self.assertRaisesRegex(TypeError, "unsupported problem/artifact pair"):
            verify_artifact(case.problem, artifact)


if __name__ == "__main__":
    unittest.main()
