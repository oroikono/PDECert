import json
from pathlib import Path

import pytest

from pdecert import (
    CallableCandidate,
    OperatorCompileError,
    Status,
    TemplateError,
    bind_symbolic_candidate,
    compile_autodiff_problem,
    dump_template,
    load_case,
    load_template,
    template_from_case,
    template_from_dict,
    template_to_dict,
    verify,
    verify_artifact,
)


def _heat_template_payload():
    return json.loads(Path("examples/heat-template.json").read_text())


def test_template_round_trip_has_no_candidate_expressions(tmp_path):
    template = load_template("examples/heat-template.json")
    payload = template_to_dict(template)

    assert payload["template_version"] == 1
    assert payload["solution_semantics"] == "classical_strong"
    assert payload["field_names"] == ["u"]
    assert "fields" not in payload
    assert "candidate_expressions" not in payload
    with pytest.raises(TypeError):
        template.domains["x"] = (0.0, 2.0)

    path = tmp_path / "template.json"
    dump_template(template, path)
    assert template_to_dict(load_template(path)) == payload


def test_symbolic_binding_preserves_exact_proof():
    template = load_template("examples/heat-template.json")
    case = bind_symbolic_candidate(template, {"u": "exp(-pi**2*t)*sin(pi*x)"})

    report = verify(case.problem, case.candidate_fields)

    assert report.status is Status.PROVED
    assert report.decision_evidence.value == "EXACT"


def test_binding_requires_exactly_the_declared_fields():
    template = load_template("examples/heat-template.json")

    with pytest.raises(TemplateError, match="missing: u"):
        bind_symbolic_candidate(template, {})
    with pytest.raises(TemplateError, match="unknown: v"):
        bind_symbolic_candidate(template, {"u": "0", "v": "0"})


def test_binding_rejects_unknown_candidate_symbols():
    template = load_template("examples/heat-template.json")

    with pytest.raises(TemplateError, match="candidate binding failed.*unknown symbol: y"):
        bind_symbolic_candidate(template, {"u": "sin(y)"})


def test_template_rejects_unknown_operator_symbols_and_unreferenced_fields():
    payload = _heat_template_payload()
    payload["pde_residuals"][0]["expression"] = "D(u, t) - mystery"
    with pytest.raises(TemplateError, match="unknown symbol: mystery"):
        template_from_dict(payload)

    payload = _heat_template_payload()
    payload["field_names"] = ["u", "v"]
    with pytest.raises(TemplateError, match=r"do not reference field\(s\): v"):
        template_from_dict(payload)


def test_template_version_one_rejects_unsupported_solution_semantics():
    payload = _heat_template_payload()
    payload["solution_semantics"] = "weak"

    with pytest.raises(TemplateError, match="classical_strong"):
        template_from_dict(payload)


def test_template_from_case_removes_and_can_restore_candidate_binding():
    original = load_case("examples/exact_heat.json")
    template = template_from_case(original)
    restored = bind_symbolic_candidate(template, original.candidate_fields)

    assert template.name == original.problem.name
    assert restored.field_names == original.field_names
    report = verify(restored.problem, restored.candidate_fields)
    assert report.status is Status.PROVED


def test_coupled_case_keeps_both_field_slots_when_converted():
    original = load_case("examples/coupled_wave.json")
    template = template_from_case(original)
    restored = bind_symbolic_candidate(template, original.candidate_fields)

    assert template.field_names == ("u", "v")
    assert restored.field_names == ("u", "v")
    assert compile_autodiff_problem(template).variables == ("x", "t")


def test_template_compiles_directly_for_callable_evaluation():
    torch = pytest.importorskip("torch")
    template = load_template("examples/heat-template.json")

    def exact(points):
        x, t = points[:, 0:1], points[:, 1:2]
        return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)

    artifact = CallableCandidate.from_mapping({"u": exact}, dtype="float64")
    report = verify_artifact(compile_autodiff_problem(template), artifact, tolerance=1e-9)

    assert report.status is Status.INCONCLUSIVE
    assert report.max_sampled_residual < 1e-9


def test_parameterized_template_binds_symbolically_but_callable_compiler_abstains():
    payload = _heat_template_payload()
    payload["variables"].append("c")
    payload["domains"]["c"] = [0.5, 2.0]
    payload["parameters"]["c"] = ["positive"]
    payload["pde_residuals"] = [{"name": "transport", "expression": "D(u, t) + c*D(u, x)"}]
    payload["conditions"] = []
    template = template_from_dict(payload)

    case = bind_symbolic_candidate(template, {"u": "sin(x-c*t)"})
    assert verify(case.problem, case.candidate_fields).status is Status.PROVED
    with pytest.raises(OperatorCompileError, match="does not yet support parameter"):
        compile_autodiff_problem(template)
