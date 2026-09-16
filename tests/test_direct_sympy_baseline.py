"""Exact residual diagnostics must never acquire whole-problem proof semantics."""

import copy
import io
import json
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import sympy as sp
from jsonschema import Draft202012Validator

from pdecert.atlas_baselines import (
    ATLAS_SYMBOLIC_BASELINE_REPORT_VERSION,
    AtlasBaselineError,
    BaselineOutcome,
    BaselineResult,
    DirectSympyBaseline,
    FixedCollocationBaseline,
    SymbolicBaselineCheck,
    SymbolicBaselineResult,
    evaluate_atlas_baseline,
)
from pdecert.cli import INPUT_ERROR, main
from pdecert.core import _run_bounded
from pdecert.templates import TemplateError, bind_symbolic_candidate, template_from_dict


ATLAS = Path("corpus/matched")
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
CALLABLE_ID = "trained-fisher-kpp-pinn-01"


@pytest.fixture
def heat_record():
    return {
        "id": "exact-heat",
        "artifact_type": "symbolic_expression",
        "artifact": {"fields": {"u": "exp(-pi**2*t)*sin(pi*x)"}},
        "template": json.loads(Path("examples/heat-template.json").read_text()),
    }


@pytest.fixture(scope="module")
def symbolic_report():
    return evaluate_atlas_baseline(ATLAS, DirectSympyBaseline())


@pytest.fixture(scope="module")
def validators():
    return {
        version: Draft202012Validator(
            json.loads(Path(f"schema/atlas-baseline-report-v{version}.schema.json").read_text())
        )
        for version in (1, 2)
    }


def _check(outcome, identifier="pde_residuals[0]", name="same name"):
    return SymbolicBaselineCheck(
        identifier,
        name,
        "D(u, x)",
        outcome,
        materialized_residual="1" if outcome == "nonzero" else "0",
        simplified_residual="1" if outcome == "nonzero" else "0",
        reason="CAS could not decide" if outcome == "undecided" else None,
    )


def test_exact_heat_checks_every_residual_and_condition_without_a_proof(heat_record):
    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert isinstance(result, SymbolicBaselineResult)
    assert result.outcome == "zero"
    assert [check.obligation_id for check in result.checks] == [
        "pde_residuals[0]",
        "conditions[0]",
        "conditions[1]",
        "conditions[2]",
    ]
    expected = heat_record["template"]["pde_residuals"] + heat_record["template"]["conditions"]
    assert [(check.constraint, check.constraint_source) for check in result.checks] == [
        (item["name"], item["expression"]) for item in expected
    ]
    payload = result.to_dict()
    assert payload["scope"] == "represented_residuals_and_conditions_only"
    for check in payload["checks"]:
        assert check["outcome"] == "zero"
        assert check["simplified_residual"] == "0"
        assert check["evidence_kind"] == "CAS_RESIDUAL_ZERO"
        assert check["evidence_level"] == "EXACT"
    assert "status" not in payload
    assert "PROVED" not in json.dumps(payload)


def test_wrong_heat_candidate_leaves_nonconstant_initial_residual_undecided(heat_record):
    heat_record["artifact"]["fields"]["u"] = "0"

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert result.outcome == "undecided"
    assert [check.outcome for check in result.checks] == ["zero", "undecided", "zero", "zero"]
    initial = result.checks[1].to_dict()
    assert initial["simplified_residual"] == "-sin(pi*x)"
    assert initial["evidence_kind"] == "ABSTENTION"
    assert initial["evidence_level"] is None


def test_finite_constant_wrong_condition_is_nonzero_and_later_checks_still_run(heat_record):
    heat_record["template"]["conditions"][0]["expression"] = "At(u, x, 0) + 1"

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert result.outcome == "nonzero"
    assert [check.outcome for check in result.checks] == ["zero", "nonzero", "zero", "zero"]
    check = result.checks[1].to_dict()
    assert check["simplified_residual"] == "1"
    assert check["evidence_kind"] == "CAS_CONSTANT_NONZERO"
    assert check["evidence_level"] == "EXACT"
    assert "witness" not in check


def test_zero_represented_residual_does_not_certify_a_singular_candidate(heat_record):
    heat_record["artifact"]["fields"]["u"] = "1/(x-1/2)"
    heat_record["template"]["pde_residuals"] = [{"name": "stationary", "expression": "D(u,t)"}]
    heat_record["template"]["conditions"] = []
    case = bind_symbolic_candidate(
        template_from_dict(heat_record["template"]), heat_record["artifact"]["fields"]
    )
    x = next(variable for variable in case.problem.variables if str(variable) == "x")
    assert case.candidate_expressions[0].subs(x, sp.Rational(1, 2)) is sp.zoo

    baseline = DirectSympyBaseline()
    result = baseline.evaluate_record(heat_record)

    assert result.outcome == "zero"
    assert baseline.configuration()["domain_checks"] is False
    assert result.to_dict()["scope"] == "represented_residuals_and_conditions_only"
    assert "PROVED" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("source", ["sin(x) + 0*u", "exp(x) + 0*u"])
def test_variable_dependent_nonzero_expressions_are_not_global_refutations(heat_record, source):
    heat_record["template"]["pde_residuals"][0]["expression"] = source
    heat_record["template"]["conditions"] = []

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert result.outcome == "undecided"
    assert "finite real nonzero constant" in result.checks[0].reason


@pytest.mark.parametrize("source", ["0.1 - 0.1", "1e-9 - 1e-9", "Float(1) - Float(1)"])
@pytest.mark.parametrize("location", ["field", "operator"])
def test_raw_float_cancellation_abstains_before_binding(heat_record, source, location):
    if location == "field":
        heat_record["artifact"]["fields"]["u"] = source
    else:
        heat_record["template"]["pde_residuals"][0]["expression"] = f"0*u + ({source})"
    with patch("pdecert.atlas_baselines.bind_symbolic_candidate") as bind:
        result = DirectSympyBaseline().evaluate_record(heat_record)

    bind.assert_not_called()
    assert len(result.checks) == 4
    assert result.outcome == "undecided"
    assert all("inexact numeric literal" in check.reason for check in result.checks)
    assert all(check.materialized_residual is None for check in result.checks)


def test_parameter_assumptions_and_rational_expressions_remain_exact(heat_record):
    template = heat_record["template"]
    template["variables"].append("c")
    template["domains"]["c"] = [0.5, 2.0]
    template["parameters"]["c"] = ["positive"]
    template["pde_residuals"] = [{"name": "transport", "expression": "D(u,t) + c*D(u,x)"}]
    template["conditions"] = [
        {"name": "positive parameter", "expression": "sqrt(c**2) - c + 0*u"},
        {"name": "exact fraction", "expression": "At(u,t,0) - sin(x)/3"},
    ]
    heat_record["artifact"]["fields"]["u"] = "Rational(1,3)*sin(x-c*t)"

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert result.outcome == "zero"
    assert len(result.checks) == 3
    assert all(check.to_dict()["evidence_level"] == "EXACT" for check in result.checks)


@pytest.mark.parametrize(
    ("source", "rendered"),
    [("sqrt(-1) + 0*u", "I"), ("1/0 + 0*u", "zoo"), ("0/0 + 0*u", "nan")],
)
def test_complex_and_nonfinite_residuals_abstain(heat_record, source, rendered):
    heat_record["template"]["pde_residuals"][0]["expression"] = source
    heat_record["template"]["conditions"] = []

    result = DirectSympyBaseline().evaluate_record(heat_record)

    check = result.checks[0]
    assert result.outcome == "undecided"
    assert check.materialized_residual == rendered
    assert check.simplified_residual is None
    assert check.to_dict()["evidence_level"] is None


def test_binding_deadline_abstains_for_every_obligation(heat_record):
    with patch(
        "pdecert.atlas_baselines._run_bounded",
        return_value=(None, "symbolic check exceeded 0.25 seconds"),
    ) as bounded:
        result = DirectSympyBaseline(symbolic_timeout=0.25).evaluate_record(heat_record)

    assert bounded.call_count == 1
    assert bounded.call_args.args[1] == 0.25
    assert len(result.checks) == 4
    assert all(check.outcome == "undecided" for check in result.checks)
    assert all("candidate binding:" in check.reason for check in result.checks)
    assert all("exceeded" in check.reason for check in result.checks)


def test_one_obligation_deadline_does_not_skip_remaining_conditions(heat_record):
    deadlines = []

    def bounded(operation, timeout):
        deadlines.append(timeout)
        if len(deadlines) == 2:
            return None, "symbolic check exceeded 0.25 seconds"
        return _run_bounded(operation, timeout)

    with patch("pdecert.atlas_baselines._run_bounded", side_effect=bounded):
        result = DirectSympyBaseline(symbolic_timeout=0.25).evaluate_record(heat_record)

    assert deadlines == [0.25] * 5
    assert [check.outcome for check in result.checks] == ["undecided", "zero", "zero", "zero"]
    assert "exceeded" in result.checks[0].reason


def test_unavailable_deadlines_in_worker_thread_abstain_without_running_binding(heat_record):
    with patch("pdecert.atlas_baselines.bind_symbolic_candidate") as bind:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(DirectSympyBaseline().evaluate_record, heat_record).result(
                timeout=2
            )

    bind.assert_not_called()
    assert len(result.checks) == 4
    assert all(check.outcome == "undecided" for check in result.checks)
    assert all("deadlines are unavailable" in check.reason for check in result.checks)


def test_operation_budget_abstains_locally_before_simplification(heat_record):
    heat_record["template"]["pde_residuals"][0]["expression"] = "sin(x) + cos(x) + 0*u"
    baseline = DirectSympyBaseline(max_expression_ops=1)

    result = baseline.evaluate_record(heat_record)

    assert [check.outcome for check in result.checks] == ["undecided", "zero", "zero", "zero"]
    assert "exceeding 1" in result.checks[0].reason
    assert result.checks[0].materialized_residual is None


@pytest.mark.parametrize("artifact_type", ["callable_model", "future_artifact"])
def test_unsupported_artifact_type_has_abstention_not_symbolic_checks(heat_record, artifact_type):
    heat_record["artifact_type"] = artifact_type

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert isinstance(result, BaselineResult)
    assert result.outcome is BaselineOutcome.UNSUPPORTED
    assert result.evidence_kind == "ABSTENTION"
    assert result.evidence_level is None
    assert "checks" not in result.to_dict()


def test_unsupported_solution_semantics_abstains_before_template_binding(heat_record):
    heat_record["template"]["solution_semantics"] = "weak"
    with patch("pdecert.atlas_baselines.bind_symbolic_candidate") as bind:
        result = DirectSympyBaseline().evaluate_record(heat_record)

    bind.assert_not_called()
    assert result.outcome is BaselineOutcome.UNSUPPORTED
    assert "classical_strong" in result.reason


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 0.0001, 3601, 1e308])
def test_invalid_symbolic_deadline_is_rejected(value):
    with pytest.raises(ValueError, match="symbolic_timeout"):
        DirectSympyBaseline(symbolic_timeout=value)


@pytest.mark.parametrize("value", [0.001, 3600])
def test_symbolic_deadline_accepts_supported_range_endpoints(value):
    assert DirectSympyBaseline(symbolic_timeout=value).symbolic_timeout == value


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="platform has no SIGALRM")
def test_oversized_deadline_is_rejected_without_changing_signal_handler():
    previous_handler = signal.getsignal(signal.SIGALRM)
    with patch("pdecert.atlas_baselines._run_bounded") as bounded:
        with pytest.raises(ValueError, match="symbolic_timeout"):
            DirectSympyBaseline(symbolic_timeout=1e308)
    bounded.assert_not_called()
    assert signal.getsignal(signal.SIGALRM) == previous_handler


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_operation_budget_is_rejected(value):
    with pytest.raises(ValueError, match="max_expression_ops"):
        DirectSympyBaseline(max_expression_ops=value)


@pytest.mark.parametrize(
    ("outcomes", "aggregate"),
    [
        (("zero", "zero"), "zero"),
        (("zero", "undecided"), "undecided"),
        (("nonzero", "undecided"), "nonzero"),
    ],
)
def test_aggregate_is_derived_from_all_checks(outcomes, aggregate):
    result = SymbolicBaselineResult(
        tuple(
            _check(outcome, f"conditions[{index}]", name=f"condition {index}")
            for index, outcome in enumerate(outcomes)
        )
    )

    assert result.outcome == aggregate
    assert result.to_dict()["outcome"] == aggregate
    assert len({check.constraint for check in result.checks}) == 2
    assert len({check.obligation_id for check in result.checks}) == 2


def test_duplicate_constraint_names_are_invalid_and_direct_binding_abstains(heat_record):
    template = heat_record["template"]
    template["conditions"][0]["name"] = template["pde_residuals"][0]["name"]
    with pytest.raises(TemplateError, match="duplicate constraint name"):
        template_from_dict(template)

    result = DirectSympyBaseline().evaluate_record(heat_record)

    assert len(result.checks) == len({check.obligation_id for check in result.checks}) == 4
    assert all(check.outcome == "undecided" for check in result.checks)
    assert all("candidate binding:" in check.reason for check in result.checks)
    assert all("duplicate constraint name" in check.reason for check in result.checks)


def test_result_rejects_missing_checks_and_duplicate_obligation_ids():
    with pytest.raises(ValueError, match="at least one"):
        SymbolicBaselineResult(())
    with pytest.raises(ValueError, match="duplicate obligation ids"):
        SymbolicBaselineResult((_check("zero"), _check("nonzero")))


@pytest.mark.parametrize(
    "changes",
    [
        {"obligation_id": "PDE"},
        {"outcome": "PROVED"},
        {"outcome": "undecided"},
        {"materialized_residual": None},
        {"simplified_residual": "1"},
        {"reason": "unexpected"},
    ],
)
def test_symbolic_check_rejects_inconsistent_evidence(changes):
    with pytest.raises(ValueError):
        replace(_check("zero"), **changes)


@pytest.mark.parametrize("defect", ["omission", "reorder", "name", "source", "id"])
def test_runner_rejects_incomplete_or_misattributed_obligation_coverage(defect):
    class IncorrectCoverage(DirectSympyBaseline):
        def evaluate_record(self, record):
            checks = list(super().evaluate_record(record).checks)
            if defect == "omission":
                checks.pop()
            elif defect == "reorder":
                checks.reverse()
            elif defect == "name":
                checks[0] = replace(checks[0], constraint="wrong name")
            elif defect == "source":
                checks[0] = replace(checks[0], constraint_source="0*u")
            else:
                checks[0] = replace(checks[0], obligation_id="conditions[99]")
            return SymbolicBaselineResult(tuple(checks))

    with pytest.raises(AtlasBaselineError, match="every represented obligation in order"):
        evaluate_atlas_baseline(ATLAS, IncorrectCoverage(), record_ids=[SYMBOLIC_ID])


@pytest.mark.parametrize("report_version", [0, 3, True, "2"])
def test_runner_rejects_invalid_report_versions(report_version):
    class InvalidVersion(DirectSympyBaseline):
        adapter_id = "external_symbolic"

    InvalidVersion.report_version = report_version
    with pytest.raises(AtlasBaselineError, match="report_version must be 1 or 2"):
        evaluate_atlas_baseline(ATLAS, InvalidVersion(), record_ids=[CALLABLE_ID])


def test_runner_rejects_symbolic_results_under_v1_envelope():
    class SymbolicV1(DirectSympyBaseline):
        adapter_id = "external_symbolic"
        report_version = 1

    with pytest.raises(
        AtlasBaselineError, match="symbolic baseline results require report_version 2"
    ):
        evaluate_atlas_baseline(ATLAS, SymbolicV1(), record_ids=[SYMBOLIC_ID])


@pytest.mark.parametrize(
    ("base", "report_version"), [(DirectSympyBaseline, 1), (FixedCollocationBaseline, 2)]
)
def test_reserved_adapters_reject_changed_report_version(base, report_version):
    class ChangedVersion(base):
        pass

    ChangedVersion.report_version = report_version
    with pytest.raises(AtlasBaselineError, match="report_version.*versioned contract"):
        evaluate_atlas_baseline(ATLAS, ChangedVersion(), record_ids=[CALLABLE_ID])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pipeline", ["simplify"]),
        ("domain_checks", True),
        ("float_policy", "round_to_exact"),
        ("include_conditions", 1),
    ],
)
def test_reserved_direct_sympy_configuration_cannot_change_silently(key, value):
    class ChangedConfiguration(DirectSympyBaseline):
        def configuration(self):
            return {**super().configuration(), key: value}

    with pytest.raises(AtlasBaselineError, match="direct_sympy metadata.*versioned contract"):
        evaluate_atlas_baseline(ATLAS, ChangedConfiguration(), record_ids=[CALLABLE_ID])


def test_reserved_direct_sympy_adapter_version_is_enforced():
    class FutureVersion(DirectSympyBaseline):
        adapter_version = 2

    with pytest.raises(AtlasBaselineError, match="direct_sympy metadata.*versioned contract"):
        evaluate_atlas_baseline(ATLAS, FutureVersion(), record_ids=[CALLABLE_ID])


def test_v2_mixed_and_all_callable_reports_remain_method_specific(symbolic_report, validators):
    assert symbolic_report["baseline_report_version"] == ATLAS_SYMBOLIC_BASELINE_REPORT_VERSION
    assert symbolic_report["evidence_policy"] == (
        "method_specific_obligation_diagnostics_no_global_verdict"
    )
    assert [row["outcome"] for row in symbolic_report["records"]] == ["zero", "unsupported"]
    validators[2].validate(symbolic_report)
    assert "PROVED" not in json.dumps(symbolic_report)

    report = evaluate_atlas_baseline(ATLAS, DirectSympyBaseline(), record_ids=[CALLABLE_ID])
    assert report["baseline_report_version"] == 2
    assert report["records"][0]["outcome"] == "unsupported"
    assert report["records"][0]["evidence_level"] is None
    validators[2].validate(report)


def test_fixed_collocation_v1_output_is_unchanged_after_a_symbolic_run(validators):
    before = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())
    evaluate_atlas_baseline(ATLAS, DirectSympyBaseline(), record_ids=[CALLABLE_ID])
    after = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())

    assert after == before
    assert after["baseline_report_version"] == 1
    assert after["evidence_policy"] == "method_specific_empirical_diagnostics_no_proof"
    assert [row["outcome"] for row in after["records"]] == ["pass", "unsupported"]
    assert set(after["records"][0]) == {
        "artifact_id",
        "artifact_type",
        "evidence_kind",
        "evidence_level",
        "evaluations",
        "max_absolute_residual",
        "outcome",
        "problem_id",
        "reason",
        "record_id",
        "witness",
    }
    validators[1].validate(after)


@pytest.mark.parametrize("version", [1, 2])
def test_public_schemas_reject_exact_evidence_on_empirical_results(version, validators):
    report = evaluate_atlas_baseline(ATLAS, FixedCollocationBaseline())
    if version == 2:
        report["baseline_report_version"] = 2
        report["evidence_policy"] = "method_specific_obligation_diagnostics_no_global_verdict"
    validators[version].validate(report)
    report["records"][0]["evidence_level"] = "EXACT"
    assert not validators[version].is_valid(report)


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("record", "outcome", "PROVED"),
        ("record", "status", "PROVED"),
        ("record", "evaluations", 1),
        ("record", "checks", []),
        ("check", "evidence_level", "EMPIRICAL"),
        ("check", "evidence_kind", "EMPIRICAL_PASS"),
        ("check", "simplified_residual", "1"),
        ("check", "materialized_residual", None),
        ("check", "obligation_id", "heat PDE"),
    ],
)
def test_v2_schema_rejects_symbolic_empirical_mixing_and_invalid_checks(
    symbolic_report, validators, target, key, value
):
    report = copy.deepcopy(symbolic_report)
    record = report["records"][0]
    recipient = record if target == "record" else record["checks"][0]
    recipient[key] = value
    assert not validators[2].is_valid(report)


@pytest.mark.parametrize(
    ("outcomes", "aggregate"),
    [
        (("zero",), "nonzero"),
        (("zero",), "undecided"),
        (("undecided",), "zero"),
        (("nonzero",), "zero"),
        (("nonzero", "undecided"), "undecided"),
    ],
)
def test_v2_schema_rejects_malformed_aggregate_outcomes(
    symbolic_report, validators, outcomes, aggregate
):
    report = copy.deepcopy(symbolic_report)
    record = report["records"][0]
    record["checks"] = [
        _check(outcome, f"conditions[{index}]").to_dict() for index, outcome in enumerate(outcomes)
    ]
    record["outcome"] = SymbolicBaselineResult(
        tuple(_check(outcome, f"conditions[{index}]") for index, outcome in enumerate(outcomes))
    ).outcome
    validators[2].validate(report)
    record["outcome"] = aggregate
    assert not validators[2].is_valid(report)


def test_v2_schema_rejects_nonzero_evidence_for_literal_zero(symbolic_report, validators):
    report = copy.deepcopy(symbolic_report)
    report["records"][0]["checks"] = [_check("nonzero").to_dict()]
    report["records"][0]["outcome"] = "nonzero"
    validators[2].validate(report)
    report["records"][0]["checks"][0]["simplified_residual"] = "0"
    assert not validators[2].is_valid(report)


def test_v1_schema_rejects_symbolic_records_even_with_v1_envelope(symbolic_report, validators):
    report = copy.deepcopy(symbolic_report)
    report["baseline_report_version"] = 1
    report["evidence_policy"] = "method_specific_empirical_diagnostics_no_proof"
    assert not validators[1].is_valid(report)


def test_cli_direct_sympy_flags_produce_valid_v2_json(tmp_path, validators):
    output = tmp_path / "direct-sympy.json"
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(
            [
                "corpus",
                "baseline",
                str(ATLAS),
                "--method",
                "direct-sympy",
                "--record",
                SYMBOLIC_ID,
                "--symbolic-timeout",
                "1.5",
                "--max-expression-ops",
                "2000",
                "--output",
                str(output),
            ]
        )

    assert code == 0
    assert stdout.getvalue() == stderr.getvalue() == ""
    report = json.loads(output.read_text())
    assert report["baseline_report_version"] == 2
    assert report["adapter"]["id"] == "direct_sympy"
    assert report["adapter"]["configuration"]["symbolic_timeout_seconds"] == 1.5
    assert report["adapter"]["configuration"]["max_expression_ops"] == 2000
    assert [row["record_id"] for row in report["records"]] == [SYMBOLIC_ID]
    validators[2].validate(report)


@pytest.mark.parametrize(
    ("method", "flag", "value"),
    [
        ("direct-sympy", "--points-per-axis", "5"),
        ("direct-sympy", "--decimal-precision", "30"),
        ("direct-sympy", "--tolerance", "1e-9"),
        ("fixed-collocation", "--symbolic-timeout", "2"),
        ("fixed-collocation", "--max-expression-ops", "10000"),
    ],
)
def test_cli_rejects_inapplicable_method_options_before_running(method, flag, value):
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("pdecert.cli.evaluate_atlas_baseline") as evaluate:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["corpus", "baseline", str(ATLAS), "--method", method, flag, value])

    evaluate.assert_not_called()
    assert code == INPUT_ERROR
    assert stdout.getvalue() == ""
    assert "do not apply" in stderr.getvalue()


@pytest.mark.parametrize("timeout", ["1e308", "0.0001"])
def test_cli_rejects_out_of_range_symbolic_deadline_before_running(timeout):
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("pdecert.cli.evaluate_atlas_baseline") as evaluate:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "corpus",
                    "baseline",
                    str(ATLAS),
                    "--method",
                    "direct-sympy",
                    "--symbolic-timeout",
                    timeout,
                ]
            )

    evaluate.assert_not_called()
    assert code == INPUT_ERROR
    assert stdout.getvalue() == ""
    assert "symbolic_timeout" in stderr.getvalue()
