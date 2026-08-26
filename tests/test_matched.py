import unittest

from experiments.adversarial_heat import build_cases
from pdecert import (
    CallableCandidate,
    EvaluationLane,
    LaneVerificationOptions,
    MatchedCase,
    SymbolicCandidate,
)


def symbolic_lane(name="symbolic"):
    experiment = build_cases()[0]
    artifact = SymbolicCandidate.from_expressions({"u": experiment.candidate})
    return EvaluationLane(name, experiment.problem, artifact)


class MatchedCaseTests(unittest.TestCase):
    def test_requires_two_named_lanes(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            MatchedCase("heat-01", ("x", "t"), ("u",), "classical", (symbolic_lane(),))

        with self.assertRaisesRegex(ValueError, "must be unique"):
            MatchedCase(
                "heat-01",
                ("x", "t"),
                ("u",),
                "classical",
                (symbolic_lane(), symbolic_lane()),
            )

    def test_rejects_mismatched_coordinates_and_fields(self):
        lane = symbolic_lane()
        with self.assertRaisesRegex(ValueError, "coordinates"):
            MatchedCase("heat-01", ("t", "x"), ("u",), "classical", (lane, symbolic_lane("b")))
        with self.assertRaisesRegex(ValueError, "fields"):
            MatchedCase("heat-01", ("x", "t"), ("v",), "classical", (lane, symbolic_lane("b")))

    def test_rejects_unsupported_problem_artifact_pair_at_the_lane_boundary(self):
        experiment = build_cases()[0]
        artifact = CallableCandidate.from_mapping({"u": lambda points: points})
        with self.assertRaisesRegex(TypeError, "unsupported problem/artifact pair"):
            EvaluationLane("callable", experiment.problem, artifact)

    def test_lane_options_reject_invalid_resource_values(self):
        with self.assertRaisesRegex(ValueError, "tolerance"):
            LaneVerificationOptions(tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            LaneVerificationOptions(samples_per_axis=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            LaneVerificationOptions(max_expression_ops=True)

    def test_symbolic_lanes_keep_separate_reports_without_aggregate_status(self):
        case = MatchedCase(
            "heat-01",
            ("x", "t"),
            ("u",),
            "classical",
            (symbolic_lane("candidate-a"), symbolic_lane("candidate-b")),
        )
        from pdecert import verify_matched_case

        report = verify_matched_case(case)
        payload = report.to_dict()
        self.assertEqual(tuple(report.reports), ("candidate-a", "candidate-b"))
        self.assertNotIn("status", payload)
        self.assertNotIn("aggregate_status", payload)
        self.assertEqual([lane["artifact_kind"] for lane in payload["lanes"]], ["symbolic"] * 2)


if __name__ == "__main__":
    unittest.main()
