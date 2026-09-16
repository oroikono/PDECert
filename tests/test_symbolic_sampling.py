import io
import json
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import sympy as sp

from pdecert import Constraint, EvidenceLevel, Problem, Status, load_case, verify
from pdecert.cli import main
from pdecert.core import _interior_points


ROOT = Path(__file__).resolve().parents[1]
LEGACY_FRACTIONS = (0.113, 0.271, 0.419, 0.613, 0.787, 0.937)
FIXTURE = ROOT / "examples/sampling_coverage.json"


class SymbolicSamplingTests(unittest.TestCase):
    def test_preserves_the_legacy_prefix_exactly(self):
        for lower, upper in ((0.0, 1.0), (-3.0, 5.0), (2.0, 3.0)):
            expected = [lower + (upper - lower) * value for value in LEGACY_FRACTIONS]
            for count in range(1, 7):
                with self.subTest(bounds=(lower, upper), count=count):
                    self.assertEqual(_interior_points(lower, upper, count), expected[:count])

    def test_larger_budgets_add_distinct_interior_points_and_keep_the_prefix(self):
        for lower, upper in ((0.0, 1.0), (-3.0, 5.0), (2.0, 3.0)):
            for count in (7, 12, 100):
                with self.subTest(bounds=(lower, upper), count=count):
                    points = _interior_points(lower, upper, count)
                    self.assertEqual(len(points), count)
                    self.assertEqual(len(set(points)), count)
                    self.assertTrue(all(lower < point < upper for point in points))
                    self.assertEqual(points, _interior_points(lower, upper, count))
                    self.assertEqual(points[:-1], _interior_points(lower, upper, count - 1))
        self.assertEqual(_interior_points(0.0, 1.0, 9)[6:], [0.5, 0.25, 0.75])

    def test_rounding_collisions_do_not_trigger_an_unbounded_uniqueness_search(self):
        lower, upper = 1.0, math.nextafter(1.0, math.inf)
        points = _interior_points(lower, upper, 100)
        self.assertEqual(len(points), 100)
        self.assertLess(len(set(points)), 100)
        self.assertTrue(all(lower <= point <= upper for point in points))

    def test_new_point_refutes_a_candidate_that_aliases_the_legacy_samples(self):
        case = load_case(FIXTURE)
        old = verify(case.problem, case.candidate_fields, samples_per_axis=6)
        expanded = verify(case.problem, case.candidate_fields, samples_per_axis=7)
        self.assertEqual(old.status, Status.INCONCLUSIVE)
        self.assertIsNone(old.decision_evidence)
        self.assertEqual(expanded.status, Status.REFUTED)
        self.assertEqual(expanded.decision_evidence, EvidenceLevel.EMPIRICAL)
        self.assertEqual(expanded.witness.constraint, "stationary PDE")
        self.assertEqual(expanded.witness.point, {"x": 0.5})
        self.assertGreater(expanded.witness.residual, 1e-9)

    def test_passing_more_samples_does_not_prove_an_undecided_identity(self):
        x = sp.symbols("x", real=True)
        problem = Problem(
            "domain-dependent identity",
            (x,),
            {x: (0.0, 1.0)},
            (Constraint("identity", sp.sqrt(x**2) - x),),
        )
        report = verify(problem, (x,), samples_per_axis=12)
        self.assertEqual(report.status, Status.INCONCLUSIVE)
        self.assertIsNone(report.decision_evidence)
        self.assertEqual(report.max_sampled_residual, 0.0)

    def test_exact_reference_retains_exact_evidence(self):
        case = load_case(ROOT / "examples/exact_heat.json")
        report = verify(case.problem, case.candidate_fields, samples_per_axis=12)
        self.assertEqual(report.status, Status.PROVED)
        self.assertEqual(report.decision_evidence, EvidenceLevel.EXACT)

    def test_invalid_sample_budgets_are_still_rejected(self):
        case = load_case(FIXTURE)
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "at least one"):
                verify(case.problem, case.candidate_fields, samples_per_axis=count)

    def test_cli_reproduces_the_sampling_budget_example(self):
        for count, expected_code, expected_status in ((6, 2, "INCONCLUSIVE"), (7, 1, "REFUTED")):
            with self.subTest(count=count):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["verify", str(FIXTURE), "--samples-per-axis", str(count)])
                report = json.loads(output.getvalue())["report"]
                self.assertEqual(code, expected_code)
                self.assertEqual(report["status"], expected_status)
                if count == 7:
                    self.assertEqual(report["decision_evidence"], "EMPIRICAL")
                    self.assertEqual(report["witness"]["point"], {"x": 0.5})
