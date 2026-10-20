"""Input admission only: synthetic adapter claims are not numerical replay evidence."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator

from pdecert.atlas_baselines import (
    AtlasBaselineError,
    BaselineOutcome,
    BaselineResult,
    BaselineWitness,
    FixedCollocationBaseline,
    evaluate_atlas_baseline,
)
from pdecert.corpus import load_cross_artifact_atlas
from pdecert.templates import template_from_dict


ATLAS = Path("corpus/matched")
SYMBOLIC_ID = "qwen3-fisher-kpp-01"


class _ExternalAdapter:
    adapter_id = "synthetic_witness_input_test"
    adapter_version = 1
    accepted_artifact_types = ("symbolic_expression",)
    accepted_solution_semantics = ("classical_strong",)

    def __init__(self, inputs, report_version, outcome=BaselineOutcome.FAIL):
        self.inputs, self.report_version, self.outcome = inputs, report_version, outcome
        self.result = None

    def configuration(self):
        return {"tolerance": 1e-9}

    def evaluate_record(self, record):
        constraint = record["template"]["pde_residuals"][0]
        witness = BaselineWitness(constraint["name"], constraint["expression"], self.inputs, 2.0)
        self.result = BaselineResult(
            self.outcome,
            (
                "NUMERICAL_THRESHOLD_EXCEEDANCE"
                if self.outcome is BaselineOutcome.FAIL
                else "EMPIRICAL_PASS"
            ),
            "EMPIRICAL",
            evaluations=1,
            max_absolute_residual=2.0 if self.outcome is BaselineOutcome.FAIL else 0.0,
            witness=witness if self.outcome is BaselineOutcome.FAIL else None,
        )
        return self.result


@pytest.fixture(params=[1, 2])
def report_version(request):
    return request.param


@pytest.fixture
def synthetic_parameters():
    """Extend an already validated Atlas in memory, without adding natural records."""
    atlas = load_cross_artifact_atlas(ATLAS)
    template = next(row for row in atlas["records"] if row["id"] == SYMBOLIC_ID)["template"]
    template["variables"].extend(["n", "a"])
    template["domains"].update(n=[1, 3], a=[-0.5, 0.5])
    template["parameters"].update(n=["integer", "positive"], a=[])
    template_from_dict(template)
    return atlas


def _assert_report(report, inputs, report_version):
    assert report["baseline_report_version"] == report_version
    assert [row["outcome"] for row in report["records"]] == ["fail", "unsupported"]
    assert report["records"][0]["witness"]["sampled_inputs"] == inputs
    schema = json.loads(
        Path(f"schema/atlas-baseline-report-v{report_version}.schema.json").read_text()
    )
    assert list(Draft202012Validator(schema).iter_errors(report)) == []


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"unknown": 0.0}, "not a declared variable"),
        ({"x": -6.01}, "outside"),
        ({"x": 6.01}, "outside"),
        ({"t": -0.01}, "outside"),
        ({"t": 2.01}, "outside"),
    ],
)
def test_rejects_unknown_and_out_of_domain_inputs(inputs, message, report_version):
    with pytest.raises(AtlasBaselineError, match=message):
        evaluate_atlas_baseline(ATLAS, _ExternalAdapter(inputs, report_version))


@pytest.mark.parametrize("inputs", [{"n": 1.5}, {"n": 0}, {"n": 4}, {"a": -0.51}, {"a": 0.51}])
def test_rejects_invalid_synthetic_parameter_inputs(inputs, synthetic_parameters, report_version):
    message = "integer-valued" if inputs == {"n": 1.5} else "outside"
    adapter = _ExternalAdapter(inputs, report_version)
    original_atlas, original_inputs = copy.deepcopy(synthetic_parameters), copy.deepcopy(inputs)
    with patch(
        "pdecert.atlas_baselines.load_cross_artifact_atlas", return_value=synthetic_parameters
    ):
        with pytest.raises(AtlasBaselineError, match=message):
            evaluate_atlas_baseline(ATLAS, adapter)
    assert synthetic_parameters == original_atlas
    assert inputs == original_inputs
    assert adapter.result.witness.to_dict()["sampled_inputs"] == original_inputs


@pytest.mark.parametrize("inputs", [{}, {"x": -6.0}, {"x": 6.0, "t": 2.0}, {"t": 0.0}])
def test_accepts_empty_subset_and_coordinate_endpoints(inputs, report_version):
    report = evaluate_atlas_baseline(ATLAS, _ExternalAdapter(inputs, report_version))
    _assert_report(report, inputs, report_version)


@pytest.mark.parametrize(
    "inputs", [{"n": 1}, {"n": 3}, {"n": 2.0}, {"n": 3.0, "a": -0.5}, {"a": 0.5}]
)
def test_accepts_synthetic_parameter_endpoints_and_integral_floats(
    inputs, synthetic_parameters, report_version
):
    with patch(
        "pdecert.atlas_baselines.load_cross_artifact_atlas", return_value=synthetic_parameters
    ):
        report = evaluate_atlas_baseline(ATLAS, _ExternalAdapter(inputs, report_version))
    _assert_report(report, inputs, report_version)


def test_raw_integer_bounds_do_not_round_outward(synthetic_parameters, report_version):
    template = synthetic_parameters["records"][0]["template"]
    template["domains"]["n"] = [2**53 + 1, 2**53 + 3]
    template_from_dict(template)
    with patch(
        "pdecert.atlas_baselines.load_cross_artifact_atlas", return_value=synthetic_parameters
    ):
        with pytest.raises(AtlasBaselineError, match="outside"):
            evaluate_atlas_baseline(ATLAS, _ExternalAdapter({"n": 2**53}, report_version))
        report = evaluate_atlas_baseline(ATLAS, _ExternalAdapter({"n": 2**53 + 1}, report_version))
    _assert_report(report, {"n": 2**53 + 1}, report_version)
    assert type(report["records"][0]["witness"]["sampled_inputs"]["n"]) is int


def test_pass_and_unsupported_rows_keep_their_contract(report_version):
    report = evaluate_atlas_baseline(
        ATLAS, _ExternalAdapter({}, report_version, BaselineOutcome.PASS)
    )
    assert [row["outcome"] for row in report["records"]] == ["pass", "unsupported"]
    assert report["records"][0]["witness"] is None
    assert report["records"][1]["witness"] is None


@pytest.mark.parametrize("case", ["native", "subset", "constant"])
def test_builtin_result_bytes_survive_runner_admission(case):
    atlas = load_cross_artifact_atlas(ATLAS)
    record = atlas["records"][0]
    if case != "native":  # Manufactured failures only; no new natural records.
        record["artifact"]["fields"]["u"] = "0" if case == "subset" else "x"
    if case == "constant":
        record["template"]["pde_residuals"][0]["expression"] = "D(u, x)"
        record["template"]["conditions"] = []
    template_from_dict(record["template"])
    adapter = FixedCollocationBaseline(points_per_axis=3)
    expected = adapter.evaluate_record(record).to_dict()
    if case != "native":
        assert expected["outcome"] == "fail"
        assert len(expected["witness"]["sampled_inputs"]) == (1 if case == "subset" else 0)
    with patch("pdecert.atlas_baselines.load_cross_artifact_atlas", return_value=atlas):
        report = evaluate_atlas_baseline(ATLAS, adapter)
    actual = {key: report["records"][0][key] for key in expected}
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
