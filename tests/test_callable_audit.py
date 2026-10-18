"""Synthetic contract checks for user-supplied callable audits.

These analytic fixtures test audit behavior; they are not trained-model or
scientific integration evidence.
"""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from pdecert import (
    CallableCandidate,
    OperatorCompileError,
    TemplateConstraint,
    TemplateError,
    audit_callable,
    load_template,
    template_to_dict,
    template_from_dict,
)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def heat_template():
    return load_template("examples/heat-template.json")


@pytest.fixture
def exact_heat(torch):
    def field(points):
        x, t = points[:, 0:1], points[:, 1:2]
        return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)

    return field


def _candidate(field, *, dtype="float64", device="cpu"):
    return CallableCandidate.from_mapping({"u": field}, dtype=dtype, device=device)


def _audit(template, artifact, **options):
    return audit_callable(
        template,
        artifact,
        candidate_id="synthetic-heat",
        provenance={"source": "analytic test fixture"},
        **options,
    )


def _record_calls(calls):
    def field(points):
        calls.append(points)
        raise AssertionError("invalid audit must reject before calling the candidate")

    return field


@pytest.mark.parametrize("bounds", [(1e39, 2e39), (1.00000001, 1.00000002), (1.0, 1.000000001)])
def test_unrepresentable_float32_grid_rejects_before_model_execution(bounds, torch):
    template = template_from_dict(
        {
            "template_version": 1,
            "name": "dtype admission",
            "solution_semantics": "classical_strong",
            "variables": ["x"],
            "domains": {"x": list(bounds)},
            "parameters": {},
            "field_names": ["u"],
            "pde_residuals": [{"name": "PDE", "expression": "D(u, x)"}],
            "conditions": [],
        }
    )
    calls = []
    with pytest.raises(ValueError, match="cannot be represented.*dtype=float32"):
        _audit(template, _candidate(_record_calls(calls), dtype="float32"))
    assert calls == []


def test_all_obligation_grids_are_admitted_before_first_model_call(torch):
    template = template_from_dict(
        {
            "template_version": 1,
            "name": "late surface rounding",
            "solution_semantics": "classical_strong",
            "variables": ["x"],
            "domains": {"x": [0.0, 0.1]},
            "parameters": {},
            "field_names": ["u"],
            "pde_residuals": [{"name": "PDE", "expression": "D(u, x)"}],
            "conditions": [{"name": "right boundary", "expression": "At(u, x, 0.1)"}],
        }
    )
    calls = []
    with pytest.raises(ValueError, match="right boundary.*cannot be represented"):
        _audit(template, _candidate(_record_calls(calls), dtype="float32"))
    assert calls == []


def test_public_import_does_not_require_or_import_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins, sys\n"
            "original_import = builtins.__import__\n"
            "def guarded_import(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'torch':\n"
            "        raise ImportError('torch intentionally unavailable')\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded_import\n"
            "import pdecert\n"
            "assert callable(pdecert.audit_callable)\n"
            "assert 'torch' not in sys.modules\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_analytic_heat_records_each_obligation_without_claiming_a_proof(heat_template, exact_heat):
    result = _audit(
        heat_template,
        _candidate(exact_heat),
        tolerance=1e-9,
        samples_per_axis=3,
        max_sample_points=18,
    )

    assert result["audit_version"] == 1
    assert result["template"] == template_to_dict(heat_template)
    assert result["candidate"] == {
        "id": "synthetic-heat",
        "field_names": ["u"],
        "provenance": {"source": "analytic test fixture"},
        "provenance_scope": "caller_declared_not_verified",
    }
    assert {
        key: result["evaluation"][key]
        for key in (
            "backend",
            "dtype",
            "device",
            "tolerance",
            "samples_per_axis",
            "max_sample_points",
        )
    } == {
        "backend": "torch",
        "dtype": "float64",
        "device": "cpu",
        "tolerance": 1e-9,
        "samples_per_axis": 3,
        "max_sample_points": 18,
    }
    assert isinstance(result["environment"], dict) and result["environment"]
    assert "status" not in result
    assert "overall_status" not in result

    obligations = result["obligations"]
    assert [item["source_obligation_id"] for item in obligations] == [
        "constraint:0",
        "constraint:1",
        "constraint:2",
        "constraint:3",
    ]
    assert [item["kind"] for item in obligations] == ["pde", "condition", "condition", "condition"]
    assert [item["fixed_coordinates"] for item in obligations] == [
        {},
        {"t": 0.0},
        {"x": 0.0},
        {"x": 1.0},
    ]
    assert [len(item["sample_points"]) for item in obligations] == [9, 3, 3, 3]
    sources = heat_template.pde_residuals + heat_template.conditions
    for item, source in zip(obligations, sources, strict=True):
        assert item["name"] == source.name
        assert item["expression"] == source.expression
        report = item["diagnostic_report"]
        assert report["status"] == "INCONCLUSIVE"
        assert report["decision_evidence"] is None
        assert report["witness"] is None
        assert report["max_sampled_residual"] < 1e-9
        assert set(report["incomplete_reasons"]) == {source.name}
        assert "finite sampling cannot prove" in report["incomplete_reasons"][source.name]
        assert len(report["evidence_events"]) == 1
        event = report["evidence_events"][0]
        assert event["obligation_id"] == "constraint:0"
        assert event["kind"] == "EMPIRICAL_PASS"
    json.dumps(result, allow_nan=False)


def test_early_pde_failure_does_not_hide_later_initial_or_boundary_failures(
    heat_template, exact_heat
):
    def perturbed(points):
        return exact_heat(points) + points[:, 1:2] + 1.0

    result = _audit(heat_template, _candidate(perturbed), tolerance=1e-9)

    assert len(result["obligations"]) == 4
    for index, item in enumerate(result["obligations"]):
        report = item["diagnostic_report"]
        assert item["source_obligation_id"] == f"constraint:{index}"
        assert report["status"] == "REFUTED"
        assert report["decision_evidence"] == "EMPIRICAL"
        assert report["witness"]["constraint"] == item["name"]
        witness_point = report["witness"]["point"]
        row = [witness_point[name] for name in heat_template.variables]
        assert row in item["sample_points"]
        for name, value in item["fixed_coordinates"].items():
            assert witness_point[name] == value
        assert report["witness"]["residual"] > 1e-3
        assert report["evidence_events"][0]["obligation_id"] == "constraint:0"
        assert report["evidence_events"][0]["witness"] == report["witness"]


def test_template_contract_rejects_duplicate_names_before_audit(heat_template):
    with pytest.raises(TemplateError, match="duplicate constraint name"):
        replace(
            heat_template,
            pde_residuals=tuple(
                replace(constraint, name="shared name")
                for constraint in heat_template.pde_residuals
            ),
            conditions=tuple(
                replace(constraint, name="shared name") for constraint in heat_template.conditions
            ),
        )


def test_supported_coupled_fields_accept_a_different_mapping_order(heat_template, torch):
    template = replace(
        heat_template,
        variables=("x",),
        domains={"x": (0.0, 1.0)},
        field_names=("u", "v"),
        pde_residuals=(
            TemplateConstraint("first equation", "D(u, x) - v"),
            TemplateConstraint("second equation", "D(v, x) + u"),
        ),
        conditions=(
            TemplateConstraint("first value", "At(u, x, 0)"),
            TemplateConstraint("second value", "At(v, x, 0) - 1"),
        ),
    )
    artifact = CallableCandidate.from_mapping(
        {
            "v": lambda points: torch.cos(points[:, 0:1]),
            "u": lambda points: torch.sin(points[:, 0:1]),
        },
        dtype="float64",
        device="cpu",
    )

    result = _audit(template, artifact, tolerance=1e-9, samples_per_axis=2, max_sample_points=6)

    assert result["candidate"]["field_names"] == ["v", "u"]
    assert [len(item["sample_points"]) for item in result["obligations"]] == [2, 2, 1, 1]
    for item in result["obligations"]:
        assert item["diagnostic_report"]["status"] == "INCONCLUSIVE"
        assert item["diagnostic_report"]["max_sampled_residual"] < 1e-9


def test_differentiable_wrapper_preserves_physical_coordinate_chain_rule(heat_template, torch):
    template = replace(
        heat_template,
        variables=("x",),
        domains={"x": (2.0, 6.0)},
        pde_residuals=(TemplateConstraint("physical curvature", "D(u, x, 2) - 3/8"),),
        conditions=(
            TemplateConstraint("physical left value", "At(u, x, 2) - 5"),
            TemplateConstraint("physical right value", "At(u, x, 6) - 8"),
        ),
    )

    class NormalizedField(torch.nn.Module):
        def forward(self, normalized_points):
            return normalized_points[:, 0:1] ** 2

    model = NormalizedField().eval()

    def physical_field(points):
        normalized_points = (points - 2.0) / 4.0
        return 3.0 * model(normalized_points) + 5.0

    result = _audit(template, _candidate(physical_field), tolerance=1e-9)

    for item in result["obligations"]:
        assert item["diagnostic_report"]["status"] == "INCONCLUSIVE"
        assert item["diagnostic_report"]["max_sampled_residual"] < 1e-9
    assert model.training is False


@pytest.mark.parametrize("failure", ["list", "wrong_shape", "exception"])
def test_malformed_callable_output_has_explicit_abstention(heat_template, torch, failure):
    def malformed(points):
        if failure == "list":
            return [0.0] * len(points)
        if failure == "wrong_shape":
            return torch.zeros((len(points), 2), dtype=points.dtype, device=points.device)
        raise RuntimeError("synthetic forward failure")

    result = _audit(heat_template, _candidate(malformed))

    expected_reason = {
        "list": "not a torch.Tensor",
        "wrong_shape": "must return shape",
        "exception": "synthetic forward failure",
    }[failure]
    for item in result["obligations"]:
        report = item["diagnostic_report"]
        assert report["status"] == "INCONCLUSIVE"
        assert report["witness"] is None
        assert expected_reason in report["incomplete_reasons"][item["name"]]
        assert report["evidence_events"][0]["kind"] == "ABSTENTION"
        assert report["evidence_events"][0]["outcome"] == "ABSTAINED"


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_residual_is_refuted_and_strict_json_safe(heat_template, torch, nonfinite):
    def nonfinite_field(points):
        return torch.full_like(points[:, 0:1], nonfinite)

    result = _audit(heat_template, _candidate(nonfinite_field))

    json.loads(json.dumps(result, allow_nan=False))
    reports = [item["diagnostic_report"] for item in result["obligations"]]
    # Constant nonfinite fields can have a zero autodiff PDE residual. The
    # boundary and initial residuals still expose their undefined values.
    for report in reports[1:]:
        assert report["status"] == "REFUTED"
        assert report["witness"]["residual"] == "undefined"
        assert report["max_sampled_residual"] == "infinity"


@pytest.mark.parametrize("expression", ["Ei(u)", "At(u, x, 0) - At(u, x, 1)"])
def test_unsupported_operator_rejects_before_candidate_execution(heat_template, expression):
    template = replace(
        heat_template,
        pde_residuals=(TemplateConstraint("unsupported operator", expression),),
        conditions=(),
    )
    calls = []

    with pytest.raises(OperatorCompileError):
        _audit(template, _candidate(_record_calls(calls)))

    assert calls == []


def test_unsupported_late_condition_rejects_before_any_candidate_execution(heat_template):
    template = replace(
        heat_template,
        conditions=heat_template.conditions
        + (TemplateConstraint("unsupported condition", "Ei(u)"),),
    )
    calls = []

    with pytest.raises(OperatorCompileError):
        _audit(template, _candidate(_record_calls(calls)))

    assert calls == []


@pytest.mark.parametrize("field_case", ["wrong", "missing", "extra"])
def test_candidate_fields_must_match_template_exactly(heat_template, field_case):
    calls = []
    field = _record_calls(calls)
    template = heat_template
    fields = {"v": field} if field_case == "wrong" else {"u": field, "v": field}
    if field_case == "missing":
        template = replace(
            heat_template,
            field_names=("u", "v"),
            pde_residuals=(TemplateConstraint("coupled residual", "u + v"),),
            conditions=(),
        )
        fields = {"u": field}
    artifact = CallableCandidate.from_mapping(fields, dtype="float64", device="cpu")

    with pytest.raises(ValueError, match="field"):
        _audit(template, artifact)

    assert calls == []


@pytest.mark.parametrize("candidate_id", ["", "   ", None, 7])
def test_candidate_id_must_be_nonempty_text(heat_template, candidate_id):
    calls = []
    with pytest.raises((TypeError, ValueError), match="candidate_id"):
        audit_callable(
            heat_template,
            _candidate(_record_calls(calls)),
            candidate_id=candidate_id,
            provenance={"source": "synthetic"},
        )
    assert calls == []


@pytest.mark.parametrize("provenance", [{}, {"": "source"}, {"source": ""}, {1: "x"}, {"x": 1}, []])
def test_provenance_must_be_a_nonempty_string_mapping(heat_template, provenance):
    calls = []
    with pytest.raises((TypeError, ValueError), match="provenance"):
        audit_callable(
            heat_template,
            _candidate(_record_calls(calls)),
            candidate_id="synthetic",
            provenance=provenance,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("tolerance", 0),
        ("tolerance", -1),
        ("tolerance", float("nan")),
        ("tolerance", float("inf")),
        ("tolerance", True),
        ("tolerance", "1e-6"),
        ("samples_per_axis", 0),
        ("samples_per_axis", -1),
        ("samples_per_axis", 1.5),
        ("samples_per_axis", True),
        ("samples_per_axis", False),
        ("samples_per_axis", "2"),
        ("max_sample_points", 0),
        ("max_sample_points", -1),
        ("max_sample_points", 1.5),
        ("max_sample_points", True),
        ("max_sample_points", False),
        ("max_sample_points", "2"),
    ],
)
def test_invalid_settings_reject_before_candidate_execution(heat_template, option, value):
    calls = []
    with pytest.raises((TypeError, ValueError), match=option):
        _audit(heat_template, _candidate(_record_calls(calls)), **{option: value})
    assert calls == []


@pytest.mark.parametrize(("dtype", "device"), [(None, "cpu"), ("float64", None)])
def test_dtype_and_device_must_be_explicit(heat_template, dtype, device):
    calls = []
    with pytest.raises((TypeError, ValueError), match="dtype|device"):
        _audit(heat_template, _candidate(_record_calls(calls), dtype=dtype, device=device))
    assert calls == []


@pytest.mark.parametrize(("count", "budget"), [(3, 17), (10**9, 100_000)])
def test_whole_audit_budget_rejects_before_any_candidate_execution(heat_template, count, budget):
    calls = []
    with pytest.raises(ValueError, match="max_sample_points|budget|sample point"):
        _audit(
            heat_template,
            _candidate(_record_calls(calls)),
            samples_per_axis=count,
            max_sample_points=budget,
        )
    assert calls == []


def test_fully_fixed_surface_counts_as_one_sample_row(heat_template, exact_heat):
    template = replace(
        heat_template,
        conditions=(TemplateConstraint("corner", "At(At(u, x, 0), t, 0)"),),
    )
    result = _audit(template, _candidate(exact_heat), samples_per_axis=3, max_sample_points=10)

    assert [len(item["sample_points"]) for item in result["obligations"]] == [9, 1]
    assert result["obligations"][1]["sample_points"] == [[0.0, 0.0]]


def test_recorded_float32_points_are_the_coordinates_actually_evaluated(heat_template, torch):
    seen = []

    def field(points):
        seen.append(points.detach().clone())
        return points[:, 0:1] * 0.0

    result = _audit(heat_template, _candidate(field, dtype="float32"), samples_per_axis=2)

    assert result["evaluation"]["dtype"] == "float32"
    assert len(seen) == len(result["obligations"])
    for points, item in zip(seen, result["obligations"], strict=True):
        assert points.dtype is torch.float32
        assert item["sample_points"] == points.cpu().tolist()
    rounded_fraction = float(torch.tensor(0.113, dtype=torch.float32).item())
    assert result["obligations"][0]["sample_points"][0] == [rounded_fraction, rounded_fraction]
    assert rounded_fraction != 0.113


@pytest.mark.parametrize("child_training", [False, True])
def test_training_module_or_training_child_rejects_before_forward(
    heat_template, torch, child_training
):
    calls = []

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1, dtype=torch.float64)

        def forward(self, points):
            calls.append(points)
            return self.layer(points)

    model = Field()
    if child_training:
        model.eval()
        model.layer.train()
    original_modes = [module.training for module in model.modules()]

    with pytest.raises(ValueError, match="training|eval"):
        _audit(heat_template, _candidate(model))

    assert calls == []
    assert [module.training for module in model.modules()] == original_modes


@pytest.mark.parametrize("context_name", ["no_grad", "inference_mode"])
def test_disabled_autograd_context_rejects_before_candidate_execution(
    heat_template, torch, context_name
):
    calls = []
    with getattr(torch, context_name)():
        with pytest.raises((RuntimeError, ValueError), match="grad|inference"):
            _audit(heat_template, _candidate(_record_calls(calls)))
    assert calls == []


@pytest.mark.parametrize("audit_dtype", ["float64", "float32"])
def test_audit_preserves_module_parameters_gradients_dtype_device_and_mode(
    heat_template, torch, audit_dtype
):
    model = torch.nn.Linear(2, 1, dtype=torch.float64).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]], dtype=torch.float64))
        model.bias.fill_(0.25)
    parameters = tuple(model.parameters())
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 7.0)
    original_values = [parameter.detach().clone() for parameter in parameters]
    original_gradients = [parameter.grad for parameter in parameters]
    original_gradient_values = [parameter.grad.clone() for parameter in parameters]
    original_modes = [module.training for module in model.modules()]
    original_options = [
        (parameter.dtype, parameter.device, parameter.requires_grad) for parameter in parameters
    ]

    result = _audit(heat_template, _candidate(model, dtype=audit_dtype))

    assert tuple(model.parameters()) == parameters
    assert [module.training for module in model.modules()] == original_modes
    assert [
        (parameter.dtype, parameter.device, parameter.requires_grad) for parameter in parameters
    ] == original_options
    for parameter, value, gradient, gradient_value in zip(
        parameters, original_values, original_gradients, original_gradient_values, strict=True
    ):
        assert torch.equal(parameter.detach(), value)
        assert parameter.grad is gradient
        assert torch.equal(parameter.grad, gradient_value)
    assert torch.is_grad_enabled()
    if audit_dtype == "float32":
        for item in result["obligations"]:
            assert item["diagnostic_report"]["status"] == "INCONCLUSIVE"
            assert item["diagnostic_report"]["evidence_events"][0]["kind"] == "ABSTENTION"
