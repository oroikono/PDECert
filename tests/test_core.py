import unittest

import sympy as sp

from experiments.adversarial_heat import build_cases
from experiments.sigs_poisson_gauss import build_probe
from pdecert import Constraint, Problem, Status, fixed_collocation_check, verify


class ProblemValidationTests(unittest.TestCase):
    def test_domains_must_match_variables(self):
        x = sp.symbols("x", real=True)
        with self.assertRaises(ValueError):
            Problem("missing domain", (x,), {}, (Constraint("zero", sp.Integer(0)),))

    def test_problem_needs_an_obligation(self):
        x = sp.symbols("x", real=True)
        with self.assertRaises(ValueError):
            Problem("empty", (x,), {x: (0.0, 1.0)}, ())


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.name: case for case in build_cases()}

    def test_exact_and_equivalent_candidates_are_proved(self):
        for name in ("exact_heat_solution", "equivalent_expression"):
            case = self.cases[name]
            self.assertEqual(verify(case.problem, (case.candidate,)).status, Status.PROVED, name)

    def test_boundary_trap_passes_pde_only_but_is_refuted(self):
        case = self.cases["pde_only_boundary_trap"]
        accepted, _ = fixed_collocation_check(
            case.problem,
            include_conditions=False,
            grid=case.baseline_grid,
        )
        report = verify(case.problem, (case.candidate,))
        self.assertTrue(accepted)
        self.assertEqual(report.status, Status.REFUTED)
        self.assertIn("condition", report.witness.constraint)

    def test_grid_alias_passes_full_grid_but_is_refuted(self):
        case = self.cases["fixed_grid_alias"]
        accepted, _ = fixed_collocation_check(
            case.problem,
            include_conditions=True,
            grid=case.baseline_grid,
        )
        self.assertTrue(accepted)
        self.assertEqual(verify(case.problem, (case.candidate,)).status, Status.REFUTED)

    def test_hidden_singularity_is_refuted(self):
        case = self.cases["hidden_singularity"]
        report = verify(case.problem, (case.candidate,))
        self.assertEqual(report.status, Status.REFUTED)
        self.assertIn("singularity", report.witness.reason)

    def test_parameter_trap_is_refuted_off_grid(self):
        case = self.cases["single_parameter_trap"]
        accepted, _ = fixed_collocation_check(
            case.problem,
            include_conditions=True,
            grid=case.baseline_grid,
        )
        self.assertTrue(accepted)
        self.assertEqual(verify(case.problem, (case.candidate,)).status, Status.REFUTED)

    def test_sub_tolerance_error_is_not_falsely_proved(self):
        case = self.cases["below_numeric_tolerance"]
        self.assertEqual(verify(case.problem, (case.candidate,)).status, Status.INCONCLUSIVE)

    def test_sigs_candidate_has_a_boundary_counterexample(self):
        problem, candidate = build_probe()
        report = verify(problem, (candidate,))
        self.assertEqual(report.status, Status.REFUTED)
        self.assertIn("boundary", report.witness.constraint)

    def test_report_is_json_serializable(self):
        case = self.cases["exact_heat_solution"]
        payload = verify(case.problem, (case.candidate,)).to_dict()
        self.assertEqual(payload["status"], "PROVED")

    def test_invalid_verification_controls_are_rejected(self):
        case = self.cases["exact_heat_solution"]
        with self.assertRaises(ValueError):
            verify(case.problem, (case.candidate,), tolerance=0)
        with self.assertRaises(ValueError):
            verify(case.problem, (case.candidate,), samples_per_axis=0)


if __name__ == "__main__":
    unittest.main()
