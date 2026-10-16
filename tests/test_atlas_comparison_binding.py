"""Source binding tests; synthesized evidence is transport data, not a scientific run."""

import builtins
import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pdecert import (
    ATLAS_BOUND_COMPARISON_VERSION,
    AtlasComparisonError,
    compare_atlas_reports,
    evaluate_atlas_baseline,
    evaluate_cross_artifact_atlas,
    load_cross_artifact_atlas,
    validate_atlas_baseline_report,
    validate_atlas_evaluation,
)
from pdecert.atlas_baselines import (
    BaselineOutcome,
    BaselineResult,
    DirectSympyBaseline,
    FixedCollocationBaseline,
)
from pdecert.cli import INPUT_ERROR, main


ATLAS = Path(__file__).resolve().parents[1] / "corpus" / "matched"
SYMBOLIC_ID = "qwen3-fisher-kpp-01"
CALLABLE_ID = "trained-fisher-kpp-pinn-01"
BOUND_SCOPE = "source_atlas_identity_and_obligation_scope_only"


def _saved_report(obligation_id="constraint:0", *, constraint=None):
    """Explicitly synthesize an abstention or refutation without evaluating a field."""

    witness = (
        None
        if constraint is None
        else {
            "constraint": constraint,
            "point": {"x": 0.0, "t": 0.0},
            "residual": 2.0,
            "reason": "Synthetic saved witness; no mathematical replay was performed.",
        }
    )
    return {
        "report_version": 1,
        "aggregation_policy_version": 1,
        "status": "INCONCLUSIVE" if witness is None else "REFUTED",
        "decision_evidence": None if witness is None else "EMPIRICAL",
        "exact_checks": {},
        "incomplete_reasons": {obligation_id: "Synthetic saved abstention."}
        if witness is None
        else {},
        "witness": witness,
        "max_sampled_residual": 0.0 if witness is None else 2.0,
        "evidence_events": [
            {
                "obligation_id": obligation_id,
                "checker": "synthetic_saved_evidence",
                "kind": "ABSTENTION" if witness is None else "EMPIRICAL_COUNTEREXAMPLE",
                "outcome": "ABSTAINED" if witness is None else "REFUTED",
                "level": None if witness is None else "EMPIRICAL",
                "detail": "Synthetic transport evidence; no candidate computation.",
                "witness": copy.deepcopy(witness),
                "bound": None,
            }
        ],
    }


def _write_pair(root, evaluation, baseline):
    paths = root / "evaluation.json", root / "baseline.json"
    for path, payload in zip(paths, (evaluation, baseline), strict=True):
        path.write_text(json.dumps(payload), encoding="utf-8")
    return paths


class AtlasComparisonBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = load_cross_artifact_atlas(ATLAS)
        cls.evaluation = evaluate_cross_artifact_atlas(ATLAS, record_ids=[SYMBOLIC_ID])
        cls.baselines = {
            version: evaluate_atlas_baseline(ATLAS, adapter)
            for version, adapter in ((1, FixedCollocationBaseline()), (2, DirectSympyBaseline()))
        }

    def pair(self, version=1, *, mixed=False):
        evaluation = copy.deepcopy(self.evaluation)
        baseline = copy.deepcopy(self.baselines[version])
        if mixed:
            record = next(row for row in self.atlas["records"] if row["id"] == CALLABLE_ID)
            evaluation["records"].append(
                {
                    "record_id": record["id"],
                    "problem_id": record["problem_id"],
                    "artifact_id": record["artifact"]["artifact_id"],
                    "artifact_type": record["artifact_type"],
                    "evaluator": "pdecert_autodiff",
                    "report": _saved_report(),
                }
            )
            evaluation["runtime"]["torch_version"] = "synthetic-saved-provenance-no-torch-run"
        else:
            baseline["records"] = [
                row for row in baseline["records"] if row["record_id"] == SYMBOLIC_ID
            ]
        return evaluation, baseline

    def bound(self, evaluation, baseline, atlas_path=ATLAS):
        return compare_atlas_reports(evaluation, baseline, atlas_path=atlas_path)

    def assert_binding_rejected(self, evaluation, baseline):
        # Native consistency and legacy joining alone intentionally permit these claims.
        validate_atlas_evaluation(evaluation)
        validate_atlas_baseline_report(baseline)
        compare_atlas_reports(evaluation, baseline)
        with self.assertRaises(AtlasComparisonError):
            self.bound(evaluation, baseline)

    def run_cli(self, evaluation_path, baseline_path, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(
                ["corpus", "compare-reports", str(evaluation_path), str(baseline_path), *arguments]
            )
        return code, output.getvalue(), errors.getvalue()

    def test_core_generated_reports_bind_both_baseline_versions_and_allow_atlas_subset(self):
        self.assertEqual(ATLAS_BOUND_COMPARISON_VERSION, 2)
        for version in (1, 2):
            with self.subTest(version=version):
                evaluation, baseline = self.pair(version)
                legacy = compare_atlas_reports(evaluation, baseline)
                result = self.bound(evaluation, baseline)
                self.assertEqual(len(result["records"]), 1)
                self.assertGreater(len(self.atlas["records"]), len(result["records"]))
                self.assertEqual(
                    result,
                    {**legacy, "comparison_version": 2, "comparison_scope": BOUND_SCOPE},
                )

    def test_omitting_atlas_or_passing_none_preserves_legacy_version_one(self):
        evaluation, baseline = self.pair(2)
        legacy = compare_atlas_reports(evaluation, baseline)
        self.assertEqual(legacy, self.bound(evaluation, baseline, atlas_path=None))
        self.assertEqual(legacy["comparison_version"], 1)
        self.assertEqual(legacy["comparison_scope"], "declared_identity_and_native_evidence_only")

    def test_bound_output_and_inputs_are_independent_nested_copies(self):
        evaluation, baseline = self.pair(2)
        original = copy.deepcopy((evaluation, baseline))
        result = self.bound(evaluation, baseline)
        result["records"][0]["pdecert"]["report"]["evidence_events"][0]["detail"] = "changed"
        result["records"][0]["baseline"]["checks"][0]["constraint"] = "changed"
        result["sources"]["baseline"]["adapter"]["configuration"]["pipeline"].append("changed")
        self.assertEqual((evaluation, baseline), original)
        result = self.bound(evaluation, baseline)
        expected = copy.deepcopy(result)
        evaluation["atlas"]["name"] = "changed input"
        baseline["records"][0]["checks"][0]["constraint"] = "changed input"
        self.assertEqual(result, expected)

    def test_saved_mixed_callable_rows_bind_without_torch_materialization_or_replay(self):
        original_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("saved binding must not import PyTorch")
            return original_import(name, *args, **kwargs)

        for version in (1, 2):
            evaluation, baseline = self.pair(version, mixed=True)
            with self.subTest(version=version), ExitStack() as stack:
                stack.enter_context(patch("builtins.__import__", side_effect=no_torch))
                for target in (
                    "pdecert.atlas_evaluation.verify",
                    "pdecert.atlas_evaluation.materialize_frozen_callable",
                    "pdecert.frozen_callable.materialize_frozen_callable",
                    "pdecert.atlas_baselines.FixedCollocationBaseline.evaluate_record",
                    "pdecert.atlas_baselines.DirectSympyBaseline.evaluate_record",
                ):
                    stack.enter_context(patch(target, side_effect=AssertionError(target)))
                result = self.bound(evaluation, baseline)
            row = next(row for row in result["records"] if row["record_id"] == CALLABLE_ID)
            self.assertEqual(row["pdecert"]["report"]["status"], "INCONCLUSIVE")
            self.assertEqual(row["baseline"]["outcome"], "unsupported")

    def test_refuted_and_inconclusive_reports_may_stop_before_full_obligation_coverage(self):
        for obligation, constraint in (
            ("constraint:0", None),
            ("constraint:0", "Fisher KPP PDE"),
            ("domain:u:x", "u domain"),
        ):
            with self.subTest(obligation=obligation, constraint=constraint):
                evaluation, baseline = self.pair()
                evaluation["records"][0]["report"] = _saved_report(
                    obligation, constraint=constraint
                )
                result = self.bound(evaluation, baseline)
                self.assertEqual(
                    len(result["records"][0]["pdecert"]["report"]["evidence_events"]), 1
                )

    def test_matching_but_forged_atlas_declarations_are_rejected(self):
        for field, value in (("name", "forged Atlas name"), ("sha256", "a" * 64)):
            with self.subTest(field=field):
                evaluation, baseline = self.pair()
                evaluation["atlas"][field] = baseline["atlas"][field] = value
                self.assert_binding_rejected(evaluation, baseline)

    def test_source_atlas_changes_invalidate_saved_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "atlas"
            shutil.copytree(ATLAS, changed)
            manifest = changed / "atlas.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["name"] = "renamed source Atlas"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            load_cross_artifact_atlas(changed)
            with self.assertRaises(AtlasComparisonError):
                self.bound(*self.pair(), atlas_path=changed)

    def test_all_report_row_identity_fields_must_match_the_actual_source(self):
        for field in ("record_id", "problem_id", "artifact_id", "artifact_type"):
            with self.subTest(field=field):
                evaluation, baseline = self.pair(mixed=field == "artifact_type")
                index = 1 if field == "artifact_type" else 0
                for payload in (evaluation, baseline):
                    payload["records"][index][field] = (
                        "symbolic_expression" if field == "artifact_type" else "forged-identity"
                    )
                if field == "artifact_type":
                    evaluation["records"][index]["evaluator"] = "pdecert_symbolic"
                self.assert_binding_rejected(evaluation, baseline)

    def test_binding_still_requires_identical_selected_record_sets(self):
        evaluation, baseline = self.pair(mixed=True)
        baseline["records"].pop()
        with self.assertRaises(AtlasComparisonError):
            self.bound(evaluation, baseline)

    def test_foreign_constraint_and_domain_event_ids_are_rejected(self):
        for obligation in ("constraint:4", "pde_residuals[0]", "domain:v:x", "domain:u:z"):
            with self.subTest(obligation=obligation):
                evaluation, baseline = self.pair()
                evaluation["records"][0]["report"] = _saved_report(obligation)
                self.assert_binding_rejected(evaluation, baseline)

    def test_callable_rows_reject_domain_event_ids(self):
        evaluation, baseline = self.pair(mixed=True)
        evaluation["records"][1]["report"] = _saved_report("domain:u:x")
        self.assert_binding_rejected(evaluation, baseline)

    def test_proved_requires_discharged_events_for_every_constraint_and_domain(self):
        for obligation in ("constraint:0", "constraint:3", "domain:u:x", "domain:u:t"):
            for mutation in ("remove", "abstain"):
                with self.subTest(obligation=obligation, mutation=mutation):
                    evaluation, baseline = self.pair()
                    events = evaluation["records"][0]["report"]["evidence_events"]
                    if mutation == "remove":
                        events[:] = [
                            event for event in events if event["obligation_id"] != obligation
                        ]
                    else:
                        event = next(
                            event for event in events if event["obligation_id"] == obligation
                        )
                        event.update(_saved_report(obligation)["evidence_events"][0])
                    self.assert_binding_rejected(evaluation, baseline)

    def test_event_witness_names_must_match_their_constraint_or_domain_scope(self):
        for obligation, name in (("constraint:0", "initial condition"), ("domain:u:x", "v domain")):
            with self.subTest(obligation=obligation):
                evaluation, baseline = self.pair()
                evaluation["records"][0]["report"] = _saved_report(obligation, constraint=name)
                self.assert_binding_rejected(evaluation, baseline)

    def test_symbolic_baseline_checks_require_complete_ordered_source_triples(self):
        for mutation in ("missing", "reordered", "id", "name", "source"):
            with self.subTest(mutation=mutation):
                evaluation, baseline = self.pair(2)
                checks = baseline["records"][0]["checks"]
                if mutation == "missing":
                    checks.pop()
                elif mutation == "reordered":
                    checks.reverse()
                else:
                    field = {
                        "id": "obligation_id",
                        "name": "constraint",
                        "source": "constraint_source",
                    }[mutation]
                    checks[0][field] = "pde_residuals[99]" if mutation == "id" else "forged source"
                self.assert_binding_rejected(evaluation, baseline)

    def test_empirical_baseline_witness_name_and_operator_must_match_a_source_pair(self):
        for mutation in (None, "name", "source", "crossed_pair"):
            with self.subTest(mutation=mutation):
                evaluation, baseline = self.pair()
                row = baseline["records"][0]
                row.update(
                    {
                        "outcome": "fail",
                        "evidence_kind": "NUMERICAL_THRESHOLD_EXCEEDANCE",
                        "max_absolute_residual": 2.0,
                        "witness": {
                            "absolute_residual": 2.0,
                            "constraint": "Fisher KPP PDE",
                            "constraint_source": "D(u, t) - D(u, x, 2) - u*(1-u)",
                            "sampled_inputs": {"x": 0.0, "t": 0.0},
                        },
                    }
                )
                if mutation == "name":
                    row["witness"]["constraint"] = "foreign constraint"
                elif mutation == "source":
                    row["witness"]["constraint_source"] = "D(u, t)"
                elif mutation == "crossed_pair":
                    row["witness"]["constraint"] = "initial condition"
                if mutation is None:
                    self.bound(evaluation, baseline)
                else:
                    self.assert_binding_rejected(evaluation, baseline)

    def test_actual_solution_semantics_must_be_accepted_unless_baseline_abstains(self):
        evaluation, baseline = self.pair()
        baseline["adapter"].update(
            {"id": "synthetic_external_adapter", "accepted_solution_semantics": ["weak"]}
        )
        self.assert_binding_rejected(evaluation, baseline)
        baseline["records"][0].update(
            BaselineResult(
                BaselineOutcome.UNSUPPORTED, "ABSTENTION", None, reason="Unsupported semantics."
            ).to_dict()
        )
        self.bound(evaluation, baseline)

    def test_full_source_atlas_is_validated_even_when_bad_record_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "atlas"
            shutil.copytree(ATLAS, changed)
            artifact = changed / "records" / CALLABLE_ID / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaises(AtlasComparisonError):
                self.bound(*self.pair(), atlas_path=changed)

    def test_invalid_atlas_argument_never_falls_back_to_legacy_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "atlas-link"
            link.symlink_to(ATLAS, target_is_directory=True)
            for path in (root / "missing", root, link, ATLAS.parent / "community"):
                with self.subTest(path=path), self.assertRaises(AtlasComparisonError):
                    self.bound(*self.pair(), atlas_path=path)

    def test_cli_atlas_option_emits_bound_reports_and_legacy_cli_stays_unchanged(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                evaluation, baseline = self.pair(version)
                paths = _write_pair(Path(directory), evaluation, baseline)
                code, output, errors = self.run_cli(*paths, "--atlas", str(ATLAS))
                self.assertEqual((code, errors), (0, ""))
                self.assertEqual(json.loads(output), self.bound(evaluation, baseline))
                code, output, errors = self.run_cli(*paths)
                self.assertEqual((code, errors), (0, ""))
                self.assertEqual(json.loads(output), compare_atlas_reports(evaluation, baseline))

    def test_cli_module_joins_saved_reports_from_an_empty_working_directory(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                evaluation, baseline = self.pair(version)
                paths = _write_pair(root, evaluation, baseline)
                working_directory = root / "empty"
                working_directory.mkdir()
                command = [
                    sys.executable,
                    "-I",
                    "-m",
                    "pdecert.cli",
                    "corpus",
                    "compare-reports",
                    *(str(path) for path in paths),
                ]
                for arguments, expected in (
                    ([], compare_atlas_reports(evaluation, baseline)),
                    (["--atlas", str(ATLAS)], self.bound(evaluation, baseline)),
                ):
                    result = subprocess.run(
                        command + arguments,
                        cwd=working_directory,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual((result.returncode, result.stderr), (0, ""))
                    self.assertEqual(json.loads(result.stdout), expected)
                destination = root / "must-not-exist.json"
                result = subprocess.run(
                    command + ["--atlas", str(root / "missing"), "-o", str(destination)],
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual((result.returncode, result.stdout), (INPUT_ERROR, ""))
                self.assertTrue(result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(destination.exists())
                self.assertEqual(list(working_directory.iterdir()), [])

    def test_cli_invalid_or_symlink_atlas_returns_64_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_pair(root, *self.pair())
            invalid = root / "invalid-atlas"
            invalid.mkdir()
            (invalid / "atlas.json").write_text(
                '{"atlas_version": 2, "atlas_version": 2}', encoding="utf-8"
            )
            link = root / "atlas-link"
            link.symlink_to(ATLAS, target_is_directory=True)
            for atlas_path in (root / "missing", invalid, link, ATLAS.parent / "community"):
                destination = root / "comparison.json"
                with self.subTest(path=atlas_path):
                    code, output, errors = self.run_cli(
                        *paths, "--atlas", str(atlas_path), "-o", str(destination)
                    )
                    self.assertEqual((code, output), (INPUT_ERROR, ""))
                    self.assertTrue(errors)
                    self.assertNotIn("Traceback", errors)
                    self.assertFalse(destination.exists())

    def test_cli_bound_output_is_new_and_existing_files_or_symlinks_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, baseline = self.pair(2)
            paths = _write_pair(root, evaluation, baseline)
            destination = root / "comparison.json"
            code, output, errors = self.run_cli(
                *paths, "--atlas", str(ATLAS), "-o", str(destination)
            )
            self.assertEqual((code, output, errors), (0, "", ""))
            self.assertEqual(json.loads(destination.read_text()), self.bound(evaluation, baseline))
            link = root / "comparison-link.json"
            link.symlink_to(destination)
            originals = {path: path.read_bytes() for path in (*paths, destination)}
            for target in (*paths, destination, link):
                with self.subTest(target=target):
                    code, output, errors = self.run_cli(
                        *paths, "--atlas", str(ATLAS), "-o", str(target)
                    )
                    self.assertEqual((code, output), (INPUT_ERROR, ""))
                    self.assertTrue(errors)
                    self.assertEqual({path: path.read_bytes() for path in originals}, originals)
            self.assertTrue(link.is_symlink())

    def test_cli_binding_failure_preserves_inputs_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, baseline = self.pair()
            for payload in (evaluation, baseline):
                payload["records"][0]["artifact_id"] = "forged artifact"
            paths = _write_pair(root, evaluation, baseline)
            originals = [path.read_bytes() for path in paths]
            destination = root / "comparison.json"
            code, output, errors = self.run_cli(
                *paths, "--atlas", str(ATLAS), "-o", str(destination)
            )
            self.assertEqual((code, output), (INPUT_ERROR, ""))
            self.assertTrue(errors)
            self.assertNotIn("Traceback", errors)
            self.assertFalse(destination.exists())
            self.assertEqual([path.read_bytes() for path in paths], originals)


if __name__ == "__main__":
    unittest.main()
