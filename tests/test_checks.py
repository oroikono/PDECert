import unittest

import sympy as sp

from examples.polynomial_checker import ExpandedPolynomialChecker
from pdecert import (
    CheckResult,
    CheckerError,
    CheckerRegistry,
    Constraint,
    EvidenceLevel,
    Problem,
    Status,
    Witness,
    default_checker_registry,
    verify,
)


class NamedRefutationChecker:
    name = "named_refutation"

    def check(self, context):
        return CheckResult(
            witness=Witness(
                constraint=context.constraints[0].name,
                point={},
                residual="policy violation",
                reason="custom checker supplied a reproducible refutation",
            ),
            witness_level=EvidenceLevel.EMPIRICAL,
        )


class UnknownObligationChecker:
    name = "unknown_obligation"

    def check(self, context):
        del context
        return CheckResult(
            proved_obligations=frozenset({"not-in-the-problem"}),
            proof_level=EvidenceLevel.EXACT,
        )


class EmpiricalProofChecker:
    name = "empirical_proof"

    def check(self, context):
        return CheckResult(
            proved_obligations=context.obligations,
            proof_level=EvidenceLevel.EMPIRICAL,
        )


class RigorousBoundChecker:
    name = "rigorous_bound"

    def check(self, context):
        return CheckResult(
            proved_obligations=context.obligations,
            proof_level=EvidenceLevel.RIGOROUS_BOUND,
        )


class UnclassifiedWitnessChecker:
    name = "unclassified_witness"

    def check(self, context):
        return CheckResult(
            witness=Witness(
                constraint=context.constraints[0].name,
                point={},
                residual=1.0,
                reason="test witness",
            )
        )


class FailingChecker:
    name = "failing"

    def check(self, context):
        del context
        raise RuntimeError("backend unavailable")


class CheckerRegistryTests(unittest.TestCase):
    def setUp(self):
        self.x = sp.symbols("x", real=True)
        self.problem = Problem(
            "undecided zero",
            (self.x,),
            {self.x: (0.0, 1.0)},
            (Constraint("residual", sp.sin(self.x) ** 2 + sp.cos(self.x) ** 2 - 1),),
        )

    def test_default_registry_has_stable_execution_order(self):
        self.assertEqual(
            default_checker_registry().names,
            ("domain", "exact_identity", "off_grid_counterexample"),
        )

    def test_registry_is_immutable_when_a_checker_is_added(self):
        original = default_checker_registry()
        extended = original.with_checker(NamedRefutationChecker())
        self.assertNotIn("named_refutation", original.names)
        self.assertEqual(extended.names[-1], "named_refutation")

    def test_checker_can_be_inserted_before_a_named_stage(self):
        registry = default_checker_registry().with_checker(
            NamedRefutationChecker(),
            before="off_grid_counterexample",
        )
        self.assertEqual(
            registry.names,
            ("domain", "exact_identity", "named_refutation", "off_grid_counterexample"),
        )

    def test_duplicate_checker_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate checker name"):
            CheckerRegistry((NamedRefutationChecker(), NamedRefutationChecker()))

    def test_unknown_insertion_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown checker insertion point"):
            default_checker_registry().with_checker(
                NamedRefutationChecker(),
                before="missing",
            )

    def test_custom_checker_participates_in_real_verification(self):
        registry = CheckerRegistry((NamedRefutationChecker(),))
        report = verify(self.problem, (self.x,), checker_registry=registry)
        self.assertEqual(report.status, Status.REFUTED)
        self.assertEqual(report.decision_evidence, EvidenceLevel.EMPIRICAL)
        self.assertIn("custom checker", report.witness.reason)

    def test_empirical_evidence_cannot_prove_obligations(self):
        with self.assertRaisesRegex(CheckerError, "exact or rigorous-bound evidence"):
            verify(
                self.problem,
                (self.x,),
                checker_registry=CheckerRegistry((EmpiricalProofChecker(),)),
            )

    def test_rigorous_bound_can_prove_declared_obligations(self):
        report = verify(
            self.problem,
            (self.x,),
            checker_registry=CheckerRegistry((RigorousBoundChecker(),)),
        )
        self.assertEqual(report.status, Status.PROVED)
        self.assertEqual(report.decision_evidence, EvidenceLevel.RIGOROUS_BOUND)

    def test_witness_requires_an_evidence_level(self):
        with self.assertRaisesRegex(CheckerError, "witness without evidence level"):
            verify(
                self.problem,
                (self.x,),
                checker_registry=CheckerRegistry((UnclassifiedWitnessChecker(),)),
            )

    def test_documented_polynomial_checker_can_prove_known_obligations(self):
        residual = sp.Add(
            self.x**2,
            2 * self.x,
            1,
            -((self.x + 1) ** 2),
            evaluate=False,
        )
        problem = Problem(
            "polynomial identity",
            (self.x,),
            {self.x: (0.0, 1.0)},
            (Constraint("expanded identity", residual),),
        )
        registry = default_checker_registry().with_checker(
            ExpandedPolynomialChecker(),
            before="exact_identity",
        )
        report = verify(problem, (self.x,), checker_registry=registry)
        self.assertEqual(report.status, Status.PROVED)
        self.assertEqual(report.exact_checks["expanded identity"], "identity")

    def test_checker_cannot_prove_an_unknown_obligation(self):
        registry = CheckerRegistry((UnknownObligationChecker(),))
        with self.assertRaisesRegex(CheckerError, "proved unknown obligation"):
            verify(self.problem, (self.x,), checker_registry=registry)

    def test_checker_failure_identifies_the_extension(self):
        registry = CheckerRegistry((FailingChecker(),))
        with self.assertRaisesRegex(CheckerError, "checker failing.*backend unavailable"):
            verify(self.problem, (self.x,), checker_registry=registry)


if __name__ == "__main__":
    unittest.main()
