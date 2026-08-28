import pytest

from pdecert import (
    CallableCandidate,
    OperatorCompileError,
    Status,
    case_from_dict,
    compile_autodiff_problem,
    load_case,
    verify_artifact,
)


def _one_dimensional_case(expression: str):
    return case_from_dict(
        {
            "schema_version": 3,
            "name": "operator compiler test",
            "variables": ["x"],
            "domains": {"x": [0.0, 1.0]},
            "parameters": {},
            "fields": {"u": "x"},
            "pde_residuals": [{"name": "operator", "expression": expression}],
            "conditions": [],
        }
    )


def test_compiler_recovers_surfaces_from_one_trusted_case():
    problem = compile_autodiff_problem(load_case("examples/exact_heat.json"))

    assert problem.variables == ("x", "t")
    assert [dict(constraint.fixed_coordinates) for constraint in problem.constraints] == [
        {},
        {"t": 0.0},
        {"x": 0.0},
        {"x": 1.0},
    ]


def test_compiled_heat_problem_preserves_empirical_evidence_semantics():
    torch = pytest.importorskip("torch")

    def exact_heat(points):
        x, t = points[:, 0:1], points[:, 1:2]
        return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)

    problem = compile_autodiff_problem(load_case("examples/exact_heat.json"))
    artifact = CallableCandidate.from_mapping({"u": exact_heat}, dtype="float64")
    report = verify_artifact(problem, artifact, tolerance=1e-9)

    assert report.status is Status.INCONCLUSIVE
    assert report.decision_evidence is None
    assert report.max_sampled_residual < 1e-9
    assert set(report.incomplete_reasons) == {
        "heat PDE",
        "initial condition",
        "left boundary",
        "right boundary",
    }


def test_compiled_heat_problem_refutes_a_perturbed_callable():
    torch = pytest.importorskip("torch")

    def perturbed_heat(points):
        x, t = points[:, 0:1], points[:, 1:2]
        exact = torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)
        return exact + 0.1 * t * x * (1 - x)

    problem = compile_autodiff_problem(load_case("examples/exact_heat.json"))
    artifact = CallableCandidate.from_mapping({"u": perturbed_heat}, dtype="float64")
    report = verify_artifact(problem, artifact, tolerance=1e-9)

    assert report.status is Status.REFUTED
    assert report.witness.constraint == "heat PDE"
    assert set(report.witness.point) == {"x", "t"}
    assert report.witness.residual > 1e-3


def test_compiler_differentiates_composite_operator_expressions():
    pytest.importorskip("torch")
    problem = compile_autodiff_problem(_one_dimensional_case("D(u**2, x) - 2*x"))
    artifact = CallableCandidate.from_mapping(
        {"u": lambda points: points[:, 0:1]},
        dtype="float64",
    )

    report = verify_artifact(problem, artifact, tolerance=1e-9)

    assert report.status is Status.INCONCLUSIVE
    assert report.max_sampled_residual < 1e-9


def test_compiler_supports_coupled_named_fields():
    torch = pytest.importorskip("torch")
    problem = compile_autodiff_problem(load_case("examples/coupled_wave.json"))

    def u_field(points):
        x, t = points[:, 0:1], points[:, 1:2]
        return torch.sin(torch.pi * x) * torch.cos(torch.pi * t)

    def v_field(points):
        x, t = points[:, 0:1], points[:, 1:2]
        return torch.cos(torch.pi * x) * torch.sin(torch.pi * t)

    artifact = CallableCandidate.from_mapping(
        {"u": u_field, "v": v_field},
        dtype="float64",
    )
    report = verify_artifact(problem, artifact, tolerance=1e-9)

    assert report.status is Status.INCONCLUSIVE
    assert report.max_sampled_residual < 1e-9


def test_compiler_rejects_parameter_variables_until_semantics_are_defined():
    case = case_from_dict(
        {
            "schema_version": 3,
            "name": "parameterized transport",
            "variables": ["x", "t", "c"],
            "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0], "c": [0.5, 2.0]},
            "parameters": {"c": ["positive"]},
            "fields": {"u": "sin(x - c*t)"},
            "pde_residuals": [{"name": "transport", "expression": "D(u, t) + c*D(u, x)"}],
            "conditions": [],
        }
    )

    with pytest.raises(OperatorCompileError, match="does not yet support parameter"):
        compile_autodiff_problem(case)


def test_compiler_rejects_incompatible_surfaces_in_one_constraint():
    case = _one_dimensional_case("At(u, x, 0) - At(u, x, 1)")

    with pytest.raises(OperatorCompileError, match="incompatible At surfaces"):
        compile_autodiff_problem(case)


def test_compiler_rejects_a_fixed_coordinate_that_escapes_at_scope():
    case = _one_dimensional_case("At(u, x, 0) - x")

    with pytest.raises(OperatorCompileError, match="outside its At expression"):
        compile_autodiff_problem(case)

    field_case = _one_dimensional_case("At(u, x, 0) - u")
    with pytest.raises(OperatorCompileError, match="outside its At expression"):
        compile_autodiff_problem(field_case)


def test_compiler_requires_field_referenced_operator_sources():
    case = _one_dimensional_case("x - x")

    with pytest.raises(OperatorCompileError, match="do not reference candidate field"):
        compile_autodiff_problem(case)


def test_compiler_abstains_before_runtime_on_unlowered_functions():
    case = _one_dimensional_case("Ei(u)")

    with pytest.raises(OperatorCompileError, match="has no callable lowering"):
        compile_autodiff_problem(case)
