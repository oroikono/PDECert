"""Synthetic saved-report joins; these fixtures are not scientific evaluations."""

import builtins
import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pdecert import (
    ATLAS_COMPARISON_VERSION,
    AtlasComparisonError,
    AtlasEvaluationOptions,
    atlas_baseline_report_sha256,
    atlas_evaluation_sha256,
    compare_atlas_reports,
    validate_atlas_baseline_report,
    validate_atlas_evaluation,
)
from pdecert.cli import INPUT_ERROR, main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
IDENTITY_FIELDS = {"record_id", "problem_id", "artifact_id", "artifact_type"}


def _baseline(version=1):
    return json.loads((FIXTURES / f"atlas-baseline-v{version}.json").read_text(encoding="utf-8"))


def _event(kind, outcome, level, *, witness=None):
    return {
        "bound": None,
        "checker": "synthetic_transport_checker",
        "detail": "Synthetic saved evidence; no candidate computation was performed.",
        "kind": kind,
        "level": level,
        "obligation_id": "pde_residuals[0]",
        "outcome": outcome,
        "witness": witness,
    }


def _synthetic_report(status, *, empirical=False):
    report = {
        "aggregation_policy_version": 1,
        "decision_evidence": None,
        "evidence_events": [],
        "exact_checks": {},
        "incomplete_reasons": {},
        "max_sampled_residual": 0.0,
        "report_version": 1,
        "status": status,
        "witness": None,
    }
    if status == "PROVED":
        report["decision_evidence"] = "EXACT"
        report["exact_checks"] = {"synthetic PDE": "synthetic unparsed certificate text"}
        report["evidence_events"] = [_event("EXACT_CERTIFICATE", "DISCHARGED", "EXACT")]
    elif status == "REFUTED":
        witness = {
            "constraint": "synthetic PDE",
            "point": {"x": 0.5, "t": "synthetic unparsed coordinate"},
            "residual": 2.0,
            "reason": "Synthetic witness; not a scientific finding.",
        }
        level = "EMPIRICAL" if empirical else "EXACT"
        report["decision_evidence"] = level
        report["witness"] = witness
        report["max_sampled_residual"] = 2.0
        report["evidence_events"] = [
            _event(
                "EMPIRICAL_COUNTEREXAMPLE" if empirical else "EXACT_CERTIFICATE",
                "REFUTED",
                level,
                witness=copy.deepcopy(witness),
            )
        ]
    else:
        reason = (
            "Synthetic finite sampled success is not proof."
            if empirical
            else ("Synthetic saved abstention; no computation was performed.")
        )
        report["incomplete_reasons"] = {"pde_residuals[0]": reason}
        report["evidence_events"] = [
            _event("EMPIRICAL_PASS", "OBSERVED_PASS", "EMPIRICAL")
            if empirical
            else _event("ABSTENTION", "ABSTAINED", None)
        ]
    return report


def _evaluation_for(baseline):
    """Build synthetic fixture rows; chosen statuses do not define comparison semantics."""

    records = []
    for row in baseline["records"]:
        callable_row = row["artifact_type"] == "callable_model"
        outcome = row["outcome"]
        status = (
            "INCONCLUSIVE"
            if callable_row or outcome in {"unsupported", "undecided"}
            else "REFUTED"
            if outcome in {"fail", "nonzero"}
            else "PROVED"
        )
        records.append(
            {
                **{key: row[key] for key in IDENTITY_FIELDS},
                "evaluator": "pdecert_autodiff" if callable_row else "pdecert_symbolic",
                "report": _synthetic_report(status, empirical=callable_row or outcome == "fail"),
            }
        )
    runtime = {
        "pdecert_version": "synthetic-comparison-test",
        "python_version": "synthetic-comparison-test",
        "sympy_version": "synthetic-comparison-test",
    }
    if any(row["artifact_type"] == "callable_model" for row in records):
        runtime["torch_version"] = "synthetic-provenance-only-no-torch-run"
    return {
        "atlas": copy.deepcopy(baseline["atlas"]),
        "evaluation_version": 1,
        "evidence_policy": "per_record_no_aggregation",
        "options": AtlasEvaluationOptions().to_dict(),
        "records": records,
        "runtime": runtime,
    }


def _pair(version=1):
    baseline = _baseline(version)
    return _evaluation_for(baseline), baseline


def _replace(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _write_pair(root, evaluation, baseline):
    evaluation_path, baseline_path = root / "evaluation.json", root / "baseline.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return evaluation_path, baseline_path


class AtlasComparisonTests(unittest.TestCase):
    def test_public_contract_is_versioned_and_uses_a_value_error(self):
        self.assertEqual(ATLAS_COMPARISON_VERSION, 1)
        self.assertTrue(issubclass(AtlasComparisonError, ValueError))

    def test_both_saved_baseline_versions_preserve_full_native_payloads(self):
        for version in (1, 2):
            with self.subTest(version=version):
                evaluation, baseline = _pair(version)
                normalized = validate_atlas_evaluation(evaluation)
                validated_baseline = validate_atlas_baseline_report(baseline)

                comparison = compare_atlas_reports(evaluation, baseline)

                self.assertEqual(
                    set(comparison),
                    {
                        "comparison_version",
                        "comparison_scope",
                        "evidence_policy",
                        "atlas",
                        "sources",
                        "records",
                    },
                )
                self.assertEqual(comparison["comparison_version"], 1)
                self.assertEqual(
                    comparison["comparison_scope"],
                    "declared_identity_and_native_evidence_only",
                )
                self.assertEqual(comparison["evidence_policy"], "per_record_no_aggregation")
                self.assertEqual(comparison["atlas"], normalized["atlas"])
                self.assertEqual(
                    comparison["sources"],
                    {
                        "evaluation": {
                            "report_sha256": atlas_evaluation_sha256(evaluation),
                            **{
                                key: normalized[key]
                                for key in (
                                    "evaluation_version",
                                    "evidence_policy",
                                    "options",
                                    "runtime",
                                )
                            },
                        },
                        "baseline": {
                            "report_sha256": atlas_baseline_report_sha256(baseline),
                            **{
                                key: validated_baseline[key]
                                for key in (
                                    "baseline_report_version",
                                    "evidence_policy",
                                    "adapter",
                                    "runtime",
                                )
                            },
                        },
                    },
                )
                evaluation_rows = {row["record_id"]: row for row in normalized["records"]}
                baseline_rows = {row["record_id"]: row for row in baseline["records"]}
                expected = []
                for record_id in sorted(evaluation_rows):
                    row, baseline_row = evaluation_rows[record_id], baseline_rows[record_id]
                    expected.append(
                        {
                            **{key: row[key] for key in IDENTITY_FIELDS},
                            "pdecert": {"evaluator": row["evaluator"], "report": row["report"]},
                            "baseline": {
                                key: value
                                for key, value in baseline_row.items()
                                if key not in IDENTITY_FIELDS
                            },
                        }
                    )
                self.assertEqual(comparison["records"], expected)
                self.assertEqual(
                    {row["pdecert"]["report"]["status"] for row in comparison["records"]},
                    {"PROVED", "REFUTED", "INCONCLUSIVE"},
                )
                self.assertNotIn("status", comparison)
                self.assertNotIn("accuracy", comparison)

    def test_join_is_sorted_while_source_digests_preserve_source_record_order(self):
        evaluation, baseline = _pair(2)
        first = compare_atlas_reports(evaluation, baseline)
        evaluation["records"].reverse()
        baseline["records"] = baseline["records"][1:] + baseline["records"][:1]

        second = compare_atlas_reports(evaluation, baseline)

        self.assertEqual(first["records"], second["records"])
        self.assertEqual(
            [row["record_id"] for row in second["records"]],
            sorted(row["record_id"] for row in evaluation["records"]),
        )
        for source in ("evaluation", "baseline"):
            self.assertNotEqual(
                first["sources"][source]["report_sha256"],
                second["sources"][source]["report_sha256"],
            )

    def test_native_disagreement_is_preserved_without_promoting_or_rejecting_evidence(self):
        for version, pdecert_status, baseline_outcome in (
            (2, "REFUTED", "zero"),
            (1, "INCONCLUSIVE", "pass"),
        ):
            with self.subTest(version=version):
                evaluation, baseline = _pair(version)
                evaluation["records"][0]["report"] = _synthetic_report(
                    pdecert_status, empirical=version == 1
                )
                record_id = evaluation["records"][0]["record_id"]
                expected_report = validate_atlas_evaluation(evaluation)["records"][0]["report"]

                result = compare_atlas_reports(evaluation, baseline)

                joined = next(row for row in result["records"] if row["record_id"] == record_id)
                self.assertEqual(joined["pdecert"]["report"], expected_report)
                self.assertEqual(joined["pdecert"]["report"]["status"], pdecert_status)
                self.assertEqual(joined["baseline"]["outcome"], baseline_outcome)
                if version == 1:
                    self.assertIsNone(joined["pdecert"]["report"]["decision_evidence"])
                    self.assertEqual(joined["baseline"]["evidence_level"], "EMPIRICAL")
                else:
                    self.assertIsNotNone(joined["pdecert"]["report"]["witness"])
                    self.assertEqual(
                        joined["baseline"]["checks"][0]["evidence_kind"], "CAS_RESIDUAL_ZERO"
                    )

    def test_evaluation_normalization_and_hashes_match_existing_consumer_contracts(self):
        evaluation, baseline = _pair()
        for key in ("symbolic_tolerance", "callable_tolerance", "symbolic_timeout"):
            evaluation["options"][key] = 1
        evaluation["records"][0]["report"]["max_sampled_residual"] = 0
        before = copy.deepcopy(evaluation)

        result = compare_atlas_reports(evaluation, baseline)

        self.assertEqual(evaluation, before)
        self.assertEqual(
            result["sources"]["evaluation"]["options"],
            validate_atlas_evaluation(evaluation)["options"],
        )
        self.assertIs(type(result["sources"]["evaluation"]["options"]["symbolic_timeout"]), float)
        self.assertEqual(
            result["sources"]["evaluation"]["report_sha256"], atlas_evaluation_sha256(evaluation)
        )
        self.assertEqual(
            result["sources"]["baseline"]["report_sha256"], atlas_baseline_report_sha256(baseline)
        )

    def test_result_and_both_inputs_are_defensive_nested_copies(self):
        evaluation, baseline = _pair(2)
        original_evaluation, original_baseline = copy.deepcopy(evaluation), copy.deepcopy(baseline)
        result = compare_atlas_reports(evaluation, baseline)
        result_before = copy.deepcopy(result)
        evaluation["atlas"]["name"] = "changed input"
        evaluation["records"][0]["report"]["evidence_events"][0]["detail"] = "changed input"
        baseline["adapter"]["configuration"]["pipeline"].append("changed input")
        baseline["records"][1]["checks"][0]["constraint"] = "changed input"
        self.assertEqual(result, result_before)

        evaluation, baseline = original_evaluation, original_baseline
        evaluation_before, baseline_before = copy.deepcopy(evaluation), copy.deepcopy(baseline)
        result = compare_atlas_reports(evaluation, baseline)
        result["atlas"]["name"] = "changed result"
        result["sources"]["evaluation"]["options"]["samples_per_axis"] = 999
        result["sources"]["evaluation"]["runtime"]["python_version"] = "changed result"
        result["sources"]["baseline"]["runtime"]["mpmath_version"] = "changed result"
        result["sources"]["baseline"]["adapter"]["configuration"]["pipeline"].append("changed")
        for row in result["records"]:
            row["pdecert"]["report"]["evidence_events"][0]["detail"] = "changed result"
            if row["pdecert"]["report"]["witness"] is not None:
                row["pdecert"]["report"]["witness"]["point"]["x"] = 100.0
            if "checks" in row["baseline"]:
                row["baseline"]["checks"][0]["constraint"] = "changed result"
        self.assertEqual(evaluation, evaluation_before)
        self.assertEqual(baseline, baseline_before)

    def test_comparison_never_replays_candidates_or_imports_torch(self):
        evaluation, baseline = _pair(2)
        baseline["records"][0]["checks"][0]["materialized_residual"] = (
            "synthetic opaque text that must never be parsed"
        )
        original_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("saved-report comparison must not import torch")
            return original_import(name, *args, **kwargs)

        forbidden = (
            "pdecert.atlas_evaluation.evaluate_cross_artifact_atlas",
            "pdecert.atlas_evaluation._evaluate_record",
            "pdecert.atlas_evaluation.verify",
            "pdecert.atlas_evaluation.verify_artifact",
            "pdecert.atlas_evaluation.bind_symbolic_candidate",
            "pdecert.atlas_evaluation.materialize_frozen_callable",
            "pdecert.atlas_evaluation.load_cross_artifact_atlas",
            "pdecert.atlas_baselines.evaluate_atlas_baseline",
            "pdecert.atlas_baselines.FixedCollocationBaseline.evaluate_record",
            "pdecert.atlas_baselines.DirectSympyBaseline.evaluate_record",
            "sympy.sympify",
            "sympy.parsing.sympy_parser.parse_expr",
        )
        with ExitStack() as stack:
            stack.enter_context(patch("builtins.__import__", side_effect=without_torch))
            for target in forbidden:
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            result = compare_atlas_reports(evaluation, baseline)
        by_id = {row["record_id"]: row for row in result["records"]}
        self.assertEqual(
            by_id["synthetic-symbolic-zero"]["baseline"]["checks"][0]["materialized_residual"],
            baseline["records"][0]["checks"][0]["materialized_residual"],
        )

    def test_empirical_witness_integer_coordinates_are_preserved_without_rounding(self):
        evaluation, baseline = _pair()
        result = compare_atlas_reports(evaluation, baseline)
        row = next(row for row in result["records"] if row["baseline"]["outcome"] == "fail")
        self.assertEqual(row["baseline"]["witness"], baseline["records"][1]["witness"])
        self.assertIs(type(row["baseline"]["witness"]["sampled_inputs"]["n"]), int)
        self.assertEqual(row["baseline"]["witness"]["sampled_inputs"]["n"], 9007199254740993)
        self.assertEqual(
            row["pdecert"]["report"]["witness"], evaluation["records"][1]["report"]["witness"]
        )

    def test_external_mixed_method_report_keeps_configuration_and_callable_empirical_pass(self):
        baseline = _baseline(2)
        baseline["adapter"] = {
            "accepted_artifact_types": ["symbolic_expression", "callable_model"],
            "accepted_solution_semantics": ["classical_strong"],
            "configuration": {
                "synthetic_settings": {"flags": [True, None, "opaque"], "seed": 9007199254740993},
                "methods": ["native_symbolic_diagnostics", "native_empirical_diagnostics"],
            },
            "id": "synthetic_external_adapter",
            "version": 7,
        }
        callable_pass = copy.deepcopy(_baseline()["records"][0])
        callable_pass.update(
            {
                "artifact_type": "callable_model",
                "artifact_id": "synthetic-callable-pass-artifact",
                "record_id": "synthetic-callable-pass",
            }
        )
        baseline["records"].append(callable_pass)
        evaluation = _evaluation_for(baseline)

        result = compare_atlas_reports(evaluation, baseline)

        self.assertEqual(result["sources"]["baseline"]["adapter"], baseline["adapter"])
        by_id = {row["record_id"]: row for row in result["records"]}
        passed = by_id["synthetic-callable-pass"]
        self.assertEqual(passed["pdecert"]["report"]["status"], "INCONCLUSIVE")
        self.assertEqual(
            passed["pdecert"]["report"]["evidence_events"][0]["kind"], "EMPIRICAL_PASS"
        )
        self.assertEqual(passed["baseline"]["outcome"], "pass")
        unsupported = by_id["synthetic-symbolic-unsupported"]
        self.assertEqual(unsupported["baseline"]["outcome"], "unsupported")
        self.assertEqual(unsupported["baseline"]["reason"], baseline["records"][3]["reason"])
        self.assertEqual(unsupported["pdecert"]["report"]["status"], "INCONCLUSIVE")

    def test_mismatching_valid_atlas_name_or_digest_is_rejected(self):
        for source in ("evaluation", "baseline"):
            for field, value in (("name", "other declared Atlas"), ("sha256", "a" * 64)):
                with self.subTest(source=source, field=field):
                    evaluation, baseline = _pair()
                    (evaluation if source == "evaluation" else baseline)["atlas"][field] = value
                    validate_atlas_evaluation(evaluation)
                    validate_atlas_baseline_report(baseline)
                    with self.assertRaises(AtlasComparisonError):
                        compare_atlas_reports(evaluation, baseline)

    def test_missing_extra_and_disjoint_record_sets_are_rejected_in_both_sources(self):
        for source in ("evaluation", "baseline"):
            for mutation in ("missing", "extra", "disjoint", "empty", "duplicate"):
                with self.subTest(source=source, mutation=mutation):
                    evaluation, baseline = _pair()
                    payload = evaluation if source == "evaluation" else baseline
                    if mutation == "missing":
                        payload["records"].pop()
                    elif mutation in {"extra", "duplicate"}:
                        row = copy.deepcopy(payload["records"][0])
                        if mutation == "extra":
                            row["record_id"] = "synthetic-extra-record"
                        payload["records"].append(row)
                    elif mutation == "disjoint":
                        for row in payload["records"]:
                            row["record_id"] += "-other"
                    else:
                        payload["records"] = []
                    if mutation in {"missing", "extra", "disjoint"}:
                        validate_atlas_evaluation(evaluation)
                        validate_atlas_baseline_report(baseline)
                    with self.assertRaises(AtlasComparisonError):
                        compare_atlas_reports(evaluation, baseline)

    def test_all_four_declared_identity_fields_must_match(self):
        for field in sorted(IDENTITY_FIELDS):
            with self.subTest(field=field):
                evaluation, baseline = _pair()
                if field == "artifact_type":
                    # An unsupported baseline row remains valid for either artifact type.
                    baseline["records"][2][field] = "symbolic_expression"
                else:
                    baseline["records"][0][field] += "-other"
                validate_atlas_evaluation(evaluation)
                validate_atlas_baseline_report(baseline)
                with self.assertRaises(AtlasComparisonError):
                    compare_atlas_reports(evaluation, baseline)

    def test_invalid_versions_policies_options_runtime_and_native_evidence_are_rejected(self):
        changes = {
            "evaluation": [
                (("evaluation_version",), True),
                (("evaluation_version",), 2),
                (("atlas", "atlas_version"), 1),
                (("evidence_policy",), "global_verdict"),
                (("options", "samples_per_axis"), 1),
                (("runtime", "torch_version"), ""),
                (("records", 0, "evaluator"), "pdecert_autodiff"),
                (("records", 0, "report", "decision_evidence"), "EMPIRICAL"),
                (("records", 1, "report", "witness"), None),
                (("records", 2, "report", "exact_checks"), {"synthetic": "0"}),
                (("records", 2, "report", "status"), "PROVED"),
            ],
            "baseline": [
                (("baseline_report_version",), True),
                (("baseline_report_version",), 3),
                (("atlas", "atlas_version"), 1),
                (("evidence_policy",), "global_verdict"),
                (("adapter", "configuration", "tolerance"), True),
                (("runtime", "python_version"), ""),
                (("records", 0, "evidence_level"), "EXACT"),
                (("records", 0, "max_absolute_residual"), 1.0),
                (("records", 1, "witness"), None),
                (("records", 2, "reason"), None),
            ],
        }
        for source, cases in changes.items():
            for path, value in cases:
                with self.subTest(source=source, path=path):
                    evaluation, baseline = _pair()
                    _replace(evaluation if source == "evaluation" else baseline, path, value)
                    with self.assertRaises(AtlasComparisonError):
                        compare_atlas_reports(evaluation, baseline)
        evaluation, baseline = _pair(2)
        baseline["records"][0]["checks"][0]["evidence_level"] = "EMPIRICAL"
        with self.assertRaises(AtlasComparisonError):
            compare_atlas_reports(evaluation, baseline)

    def test_saved_native_evidence_contradictions_cannot_be_hidden_by_a_join(self):
        evaluation, baseline = _pair()
        evaluation["records"][2]["report"] = copy.deepcopy(evaluation["records"][0]["report"])
        with self.assertRaisesRegex(AtlasComparisonError, "cannot be PROVED"):
            compare_atlas_reports(evaluation, baseline)
        evaluation, baseline = _pair()
        evaluation["records"][0]["report"]["evidence_events"].append(
            copy.deepcopy(evaluation["records"][1]["report"]["evidence_events"][0])
        )
        with self.assertRaisesRegex(AtlasComparisonError, "refuting evidence"):
            compare_atlas_reports(evaluation, baseline)

    def test_non_json_values_in_recognized_fields_are_rejected_without_coercion(self):
        for value in (float("nan"), float("inf"), (1, 2), {1}, b"bytes", {1: "key"}):
            with self.subTest(value=value):
                evaluation, baseline = _pair()
                evaluation["options"]["symbolic_tolerance"] = value
                with self.assertRaises(AtlasComparisonError):
                    compare_atlas_reports(evaluation, baseline)
                evaluation, baseline = _pair()
                baseline["adapter"]["id"] = "synthetic_external_adapter"
                baseline["adapter"]["configuration"] = {"synthetic_settings": value}
                with self.assertRaises(AtlasComparisonError):
                    compare_atlas_reports(evaluation, baseline)
        evaluation, baseline = _pair()
        evaluation["records"][0]["artifact_type"] = []
        with self.assertRaises(AtlasComparisonError):
            compare_atlas_reports(evaluation, baseline)
        evaluation, baseline = _pair()
        evaluation["runtime"]["python_version"] = {"cycle": None}
        evaluation["runtime"]["python_version"]["cycle"] = evaluation
        with self.assertRaises(AtlasComparisonError):
            compare_atlas_reports(evaluation, baseline)

    def test_non_json_inputs_unknown_fields_and_excessive_nesting_are_reported_as_input_errors(
        self,
    ):
        for invalid in (None, [], "{}", True, {1: "nonstring key"}):
            for source in ("evaluation", "baseline"):
                with self.subTest(source=source, invalid=invalid):
                    evaluation, baseline = _pair()
                    with self.assertRaises(AtlasComparisonError):
                        compare_atlas_reports(
                            invalid if source == "evaluation" else evaluation,
                            invalid if source == "baseline" else baseline,
                        )
        for source in ("evaluation", "baseline"):
            for value in (float("nan"), float("inf"), (1, 2), {1}, b"bytes", {1: "key"}):
                with self.subTest(source=source, value=value):
                    evaluation, baseline = _pair()
                    (evaluation if source == "evaluation" else baseline)["extra"] = value
                    with self.assertRaises(AtlasComparisonError):
                        compare_atlas_reports(evaluation, baseline)
        evaluation, baseline = _pair()
        evaluation["records"][0]["report"]["exact_checks"]["synthetic"] = b"bytes"
        with self.assertRaises(AtlasComparisonError):
            compare_atlas_reports(evaluation, baseline)
        nested = None
        for _ in range(1500):
            nested = [nested]
        evaluation, baseline = _pair()
        baseline["adapter"]["id"] = "synthetic_external_adapter"
        baseline["adapter"]["configuration"] = {"nested": nested}
        with self.assertRaises(AtlasComparisonError):
            compare_atlas_reports(evaluation, baseline)


class AtlasComparisonCliTests(unittest.TestCase):
    def _run(self, evaluation_path, baseline_path, *extra):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(
                ["corpus", "compare-reports", str(evaluation_path), str(baseline_path), *extra]
            )
        return code, output.getvalue(), errors.getvalue()

    def test_stdout_is_the_complete_join_and_success_does_not_mean_a_proof(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                evaluation, baseline = _pair(version)
                evaluation_path, baseline_path = _write_pair(Path(directory), evaluation, baseline)
                code, output, errors = self._run(evaluation_path, baseline_path)
                self.assertEqual(code, 0)
                self.assertEqual(errors, "")
                result = json.loads(output)
                self.assertEqual(result, compare_atlas_reports(evaluation, baseline))
                self.assertIn(
                    "REFUTED", {row["pdecert"]["report"]["status"] for row in result["records"]}
                )
                self.assertIn(
                    "INCONCLUSIVE",
                    {row["pdecert"]["report"]["status"] for row in result["records"]},
                )

    def test_output_option_creates_a_complete_strict_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, baseline = _pair(2)
            evaluation_path, baseline_path = _write_pair(root, evaluation, baseline)
            destination = root / "comparison.json"
            code, output, errors = self._run(evaluation_path, baseline_path, "-o", str(destination))
            self.assertEqual((code, output, errors), (0, "", ""))
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                compare_atlas_reports(evaluation, baseline),
            )

    def test_mismatched_atlases_identities_and_record_sets_do_not_create_output(self):
        for mutation in ("atlas", "identity", "records"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evaluation, baseline = _pair()
                if mutation == "atlas":
                    baseline["atlas"]["sha256"] = "b" * 64
                elif mutation == "identity":
                    baseline["records"][0]["artifact_id"] = "synthetic-other-artifact"
                else:
                    baseline["records"].pop()
                evaluation_path, baseline_path = _write_pair(root, evaluation, baseline)
                destination = root / "comparison.json"
                code, output, errors = self._run(
                    evaluation_path, baseline_path, "-o", str(destination)
                )
                self.assertEqual(code, INPUT_ERROR)
                self.assertEqual(output, "")
                self.assertTrue(errors)
                self.assertNotIn("Traceback", errors)
                self.assertFalse(destination.exists())

    def test_invalid_saved_inputs_return_64_and_leave_no_output(self):
        invalid_documents = (
            b"{",
            b"{}",
            b"\xff",
            b'{"evaluation_version": 1, "evaluation_version": 1}',
            b'{"baseline_report_version": 1, "baseline_report_version": 1}',
            b'{"nested": NaN}',
            ("[" * 1500 + "0" + "]" * 1500).encode(),
        )
        for source in ("evaluation", "baseline"):
            for document in invalid_documents:
                with (
                    self.subTest(source=source, prefix=document[:40]),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    evaluation_path, baseline_path = _write_pair(root, *_pair())
                    (evaluation_path if source == "evaluation" else baseline_path).write_bytes(
                        document
                    )
                    destination = root / "comparison.json"
                    code, output, errors = self._run(
                        evaluation_path, baseline_path, "-o", str(destination)
                    )
                    self.assertEqual(code, 64)
                    self.assertEqual(output, "")
                    self.assertTrue(errors)
                    self.assertNotIn("Traceback", errors)
                    self.assertFalse(destination.exists())

    def test_missing_directory_and_symlink_inputs_are_rejected_on_either_side(self):
        for source in ("evaluation", "baseline"):
            for kind in ("missing", "directory", "symlink"):
                with (
                    self.subTest(source=source, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    evaluation_path, baseline_path = _write_pair(root, *_pair())
                    source_path = evaluation_path if source == "evaluation" else baseline_path
                    invalid = root / "missing.json" if kind == "missing" else root
                    if kind == "symlink":
                        invalid = root / "link.json"
                        invalid.symlink_to(source_path)
                    paths = (
                        (invalid, baseline_path)
                        if source == "evaluation"
                        else (evaluation_path, invalid)
                    )
                    code, output, errors = self._run(*paths)
                    self.assertEqual(code, INPUT_ERROR)
                    self.assertEqual(output, "")
                    self.assertIn("regular file", errors)

    def test_saved_report_size_limits_are_used_before_comparison(self):
        for module, constant in (
            ("pdecert.atlas_evaluation", "ATLAS_EVALUATION_MAX_BYTES"),
            ("pdecert.atlas_baseline_reports", "ATLAS_BASELINE_MAX_BYTES"),
        ):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evaluation_path, baseline_path = _write_pair(root, *_pair())
                destination = root / "comparison.json"
                with patch(f"{module}.{constant}", 10):
                    code, output, errors = self._run(
                        evaluation_path, baseline_path, "-o", str(destination)
                    )
                self.assertEqual(code, INPUT_ERROR)
                self.assertEqual(output, "")
                self.assertTrue(errors)
                self.assertFalse(destination.exists())

    def test_existing_output_sources_directories_and_symlinks_are_never_overwritten(self):
        for kind in ("file", "evaluation", "baseline", "directory", "symlink", "dangling"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evaluation_path, baseline_path = _write_pair(root, *_pair())
                old_evaluation = evaluation_path.read_bytes()
                old_baseline = baseline_path.read_bytes()
                existing = root / "existing.json"
                existing.write_text("existing user content\n", encoding="utf-8")
                destination = {
                    "file": existing,
                    "evaluation": evaluation_path,
                    "baseline": baseline_path,
                    "directory": root,
                }.get(kind, root / "link.json")
                if kind in {"symlink", "dangling"}:
                    destination.symlink_to(existing if kind == "symlink" else root / "missing")
                code, output, errors = self._run(
                    evaluation_path, baseline_path, "--output", str(destination)
                )
                self.assertEqual(code, INPUT_ERROR)
                self.assertEqual(output, "")
                self.assertTrue(errors)
                self.assertNotIn("Traceback", errors)
                self.assertEqual(existing.read_text(encoding="utf-8"), "existing user content\n")
                self.assertEqual(evaluation_path.read_bytes(), old_evaluation)
                self.assertEqual(baseline_path.read_bytes(), old_baseline)
                if kind in {"symlink", "dangling"}:
                    self.assertTrue(destination.is_symlink())
                    self.assertFalse((root / "missing").exists())

    def test_output_creation_error_returns_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path, baseline_path = _write_pair(root, *_pair())
            destination = root / "missing-parent" / "comparison.json"
            code, output, errors = self._run(evaluation_path, baseline_path, "-o", str(destination))
            self.assertEqual(code, INPUT_ERROR)
            self.assertEqual(output, "")
            self.assertTrue(errors)
            self.assertNotIn("Traceback", errors)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
