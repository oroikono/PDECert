import math

import pytest
import sympy as sp

torch = pytest.importorskip("torch")

from pdecert import (  # noqa: E402
    AutodiffConstraint,
    AutodiffProblem,
    CallableCandidate,
    Constraint,
    EvaluationLane,
    LaneVerificationOptions,
    MatchedCase,
    Problem,
    Status,
    SymbolicCandidate,
    verify_artifact,
    verify_matched_case,
)


def heat_problem():
    return AutodiffProblem(
        name="callable heat equation",
        variables=("x", "t"),
        domains={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        pde_residuals=(
            AutodiffConstraint(
                "heat PDE",
                lambda evaluation: evaluation.derivative("u", "t")
                - evaluation.derivative("u", "x", order=2),
            ),
        ),
        conditions=(
            AutodiffConstraint(
                "initial condition",
                lambda evaluation: evaluation.field("u")
                - torch.sin(torch.pi * evaluation.coordinate("x")),
                fixed_coordinates={"t": 0.0},
            ),
            AutodiffConstraint(
                "left boundary",
                lambda evaluation: evaluation.field("u"),
                fixed_coordinates={"x": 0.0},
            ),
            AutodiffConstraint(
                "right boundary",
                lambda evaluation: evaluation.field("u"),
                fixed_coordinates={"x": 1.0},
            ),
        ),
    )


def exact_heat(points):
    x = points[:, 0:1]
    t = points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def perturbed_heat(points):
    x = points[:, 0:1]
    t = points[:, 1:2]
    return exact_heat(points) + 0.1 * t * x * (1 - x)


def test_exact_callable_is_inconclusive_not_falsely_proved():
    artifact = CallableCandidate.from_mapping({"u": exact_heat}, dtype="float64")
    report = verify_artifact(heat_problem(), artifact, tolerance=1e-9)
    assert report.status is Status.INCONCLUSIVE
    assert report.witness is None
    assert report.max_sampled_residual < 1e-9
    assert set(report.incomplete_reasons) == {
        "heat PDE",
        "initial condition",
        "left boundary",
        "right boundary",
    }
    assert all(
        "finite sampling cannot prove" in reason for reason in report.incomplete_reasons.values()
    )


def test_perturbed_callable_is_refuted_with_a_concrete_point():
    artifact = CallableCandidate.from_mapping({"u": perturbed_heat}, dtype="float64")
    report = verify_artifact(heat_problem(), artifact, tolerance=1e-9)
    assert report.status is Status.REFUTED
    assert report.witness.constraint == "heat PDE"
    assert set(report.witness.point) == {"x", "t"}
    assert report.witness.residual > 1e-3


def symbolic_heat_lane():
    x, t = sp.symbols("x t", real=True)
    candidate = sp.exp(-(sp.pi**2) * t) * sp.sin(sp.pi * x)
    problem = Problem(
        name="symbolic heat equation",
        variables=(x, t),
        domains={x: (0.0, 1.0), t: (0.0, 1.0)},
        pde_residuals=(Constraint("heat PDE", sp.diff(candidate, t) - sp.diff(candidate, x, 2)),),
        conditions=(
            Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
            Constraint("left boundary", candidate.subs(x, 0)),
            Constraint("right boundary", candidate.subs(x, 1)),
        ),
    )
    return EvaluationLane(
        "symbolic",
        problem,
        SymbolicCandidate.from_expressions({"u": candidate}),
    )


def matched_heat_case(callable_field):
    return MatchedCase(
        case_id="heat-classical-01",
        coordinate_names=("x", "t"),
        field_names=("u",),
        solution_semantics="classical",
        lanes=(
            symbolic_heat_lane(),
            EvaluationLane(
                "callable",
                heat_problem(),
                CallableCandidate.from_mapping({"u": callable_field}, dtype="float64"),
            ),
        ),
    )


def test_matched_case_preserves_exact_and_empirical_results_per_lane():
    report = verify_matched_case(
        matched_heat_case(exact_heat),
        options={
            "symbolic": LaneVerificationOptions(tolerance=1e-9),
            "callable": LaneVerificationOptions(tolerance=1e-9),
        },
    )
    assert report.reports["symbolic"].status is Status.PROVED
    assert report.reports["callable"].status is Status.INCONCLUSIVE
    assert [lane.artifact_kind for lane in report.lanes] == ["symbolic", "callable"]


def test_matched_case_can_refute_one_artifact_without_relabeling_the_other():
    report = verify_matched_case(matched_heat_case(perturbed_heat))
    assert report.reports["symbolic"].status is Status.PROVED
    assert report.reports["callable"].status is Status.REFUTED
    assert report.reports["callable"].witness.constraint == "heat PDE"


def test_matched_case_rejects_symbolic_limits_for_callable_lane():
    with pytest.raises(ValueError, match="do not apply to callable"):
        verify_matched_case(
            matched_heat_case(exact_heat),
            options={"callable": LaneVerificationOptions(symbolic_timeout=1.0)},
        )


def test_torch_module_parameter_dtype_is_inferred():
    class LinearField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.layer.weight.fill_(1.0)

        def forward(self, points):
            return self.layer(points)

    problem = AutodiffProblem(
        name="module derivative",
        variables=("x",),
        domains={"x": (0.0, 1.0)},
        pde_residuals=(
            AutodiffConstraint(
                "unit derivative",
                lambda evaluation: evaluation.derivative("u", "x") - 1.0,
            ),
        ),
    )
    artifact = CallableCandidate.from_mapping({"u": LinearField()})
    report = verify_artifact(problem, artifact)
    assert report.status is Status.INCONCLUSIVE
    assert report.max_sampled_residual == 0.0


def test_fixed_coordinate_condition_is_evaluated_on_boundary():
    problem = AutodiffProblem(
        name="boundary-only",
        variables=("x", "t"),
        domains={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        pde_residuals=(),
        conditions=(
            AutodiffConstraint(
                "left boundary",
                lambda evaluation: evaluation.field("u"),
                fixed_coordinates={"x": 0.0},
            ),
        ),
    )
    artifact = CallableCandidate.from_mapping(
        {"u": lambda points: points[:, 0:1] + 0.25},
        dtype="float64",
    )
    report = verify_artifact(problem, artifact, tolerance=1e-9)
    assert report.status is Status.REFUTED
    assert report.witness.point["x"] == 0.0
    assert math.isclose(report.witness.residual, 0.25)


def test_nonfinite_callable_output_is_refuted():
    problem = AutodiffProblem(
        name="nonfinite",
        variables=("x",),
        domains={"x": (0.0, 1.0)},
        pde_residuals=(
            AutodiffConstraint("finite field", lambda evaluation: evaluation.field("u")),
        ),
    )
    artifact = CallableCandidate.from_mapping(
        {"u": lambda points: points[:, 0:1] * torch.nan},
        dtype="float64",
    )
    report = verify_artifact(problem, artifact)
    assert report.status is Status.REFUTED
    assert report.witness.residual == "undefined"


def test_problem_rejects_unknown_or_out_of_domain_fixed_coordinates():
    with pytest.raises(ValueError, match="unknown variables"):
        AutodiffProblem(
            "bad surface",
            ("x",),
            {"x": (0.0, 1.0)},
            (AutodiffConstraint("bad", lambda evaluation: evaluation, {"t": 0.0}),),
        )
    with pytest.raises(ValueError, match="outside its domain"):
        AutodiffProblem(
            "bad surface",
            ("x",),
            {"x": (0.0, 1.0)},
            (AutodiffConstraint("bad", lambda evaluation: evaluation, {"x": 2.0}),),
        )
