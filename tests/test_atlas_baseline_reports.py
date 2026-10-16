"""Synthetic transport/consistency tests; these fixtures are not scientific runs."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from pdecert import (
    ATLAS_BASELINE_MAX_BYTES,
    AtlasBaselineError,
    atlas_baseline_report_sha256,
    load_atlas_baseline_report,
    validate_atlas_baseline_report,
)
from pdecert.atlas_baselines import (
    BaselineOutcome,
    BaselineResult,
    BaselineWitness,
    DirectSympyBaseline,
    FixedCollocationBaseline,
    SymbolicBaselineCheck,
    SymbolicBaselineResult,
)
from pdecert.cli import INPUT_ERROR, main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(version=1):
    return json.loads((FIXTURES / f"atlas-baseline-v{version}.json").read_text(encoding="utf-8"))


def _replace(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return payload


class AtlasBaselineReportTests(unittest.TestCase):
    def assert_invalid_changes(self, version, changes):
        for path, value in changes:
            with self.subTest(version=version, path=path, value=value):
                payload = _replace(_fixture(version), path, value)
                with self.assertRaises(AtlasBaselineError):
                    validate_atlas_baseline_report(payload)

    def test_synthetic_fixtures_match_public_schemas_and_builtin_metadata(self):
        for version, adapter in ((1, FixedCollocationBaseline()), (2, DirectSympyBaseline())):
            with self.subTest(version=version):
                payload = _fixture(version)
                schema_path = ROOT / "schema" / f"atlas-baseline-report-v{version}.schema.json"
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                self.assertEqual(list(Draft202012Validator(schema).iter_errors(payload)), [])
                self.assertEqual(payload["adapter"]["configuration"], adapter.configuration())
                self.assertEqual(payload["atlas"]["sha256"], "0" * 64)
                self.assertIn("Synthetic", payload["atlas"]["name"])
                self.assertEqual(validate_atlas_baseline_report(payload), payload)
                self.assertEqual(
                    load_atlas_baseline_report(FIXTURES / f"atlas-baseline-v{version}.json"),
                    payload,
                )

    def test_loaded_results_agree_with_dataclass_contracts(self):
        for row in validate_atlas_baseline_report(_fixture())["records"]:
            witness = BaselineWitness(**row["witness"]) if row["witness"] else None
            result = BaselineResult(
                outcome=BaselineOutcome(row["outcome"]),
                evidence_kind=row["evidence_kind"],
                evidence_level=row["evidence_level"],
                evaluations=row["evaluations"],
                max_absolute_residual=row["max_absolute_residual"],
                reason=row["reason"],
                witness=witness,
            )
            self.assertEqual(result.to_dict(), {key: row[key] for key in result.to_dict()})
        for row in validate_atlas_baseline_report(_fixture(2))["records"][:3]:
            checks = tuple(
                SymbolicBaselineCheck(
                    **{
                        key: value
                        for key, value in check.items()
                        if not key.startswith("evidence_")
                    }
                )
                for check in row["checks"]
            )
            result = SymbolicBaselineResult(checks)
            self.assertEqual(result.to_dict(), {key: row[key] for key in result.to_dict()})

    def test_validation_is_defensive_and_does_not_replay_expressions(self):
        payload = _fixture(2)
        payload["records"][0]["checks"][0]["materialized_residual"] = "synthetic unparsed text"
        before = copy.deepcopy(payload)
        with (
            patch.object(FixedCollocationBaseline, "evaluate_record", side_effect=AssertionError),
            patch.object(DirectSympyBaseline, "evaluate_record", side_effect=AssertionError),
        ):
            normalized = validate_atlas_baseline_report(payload)
            atlas_baseline_report_sha256(payload)
        self.assertEqual(payload, before)
        normalized["adapter"]["configuration"]["pipeline"].append("mutated")
        normalized["records"][0]["checks"][0]["outcome"] = "nonzero"
        self.assertEqual(payload, before)
        payload["records"][1]["checks"][0]["constraint"] = "changed input"
        self.assertNotEqual(normalized["records"][1], payload["records"][1])

    def test_missing_and_unknown_fields_are_rejected_at_every_structural_level(self):
        paths = [(), ("adapter",), ("atlas",), ("runtime",), ("records", 1)]
        paths.append(("records", 1, "witness"))
        for path in paths:
            for unknown in (False, True):
                with self.subTest(path=path, unknown=unknown):
                    payload = _fixture()
                    target = payload
                    for key in path:
                        target = target[key]
                    if unknown:
                        target["unrecognized"] = None
                    else:
                        del target[next(iter(target))]
                    with self.assertRaises(AtlasBaselineError):
                        validate_atlas_baseline_report(payload)
        self.assert_invalid_changes(2, [(("records", 0, "checks", 0, "extra"), None)])

    def test_versions_identities_and_counts_reject_booleans_and_empty_values(self):
        self.assert_invalid_changes(
            1,
            [
                (("baseline_report_version",), True),
                (("baseline_report_version",), 3),
                (("atlas", "atlas_version"), True),
                (("atlas", "atlas_version"), 1),
                (("atlas", "sha256"), "A" * 64),
                (("atlas", "name"), " "),
                (("adapter", "version"), True),
                (("adapter", "id"), " "),
                (("adapter", "accepted_artifact_types"), ["symbolic_expression"] * 2),
                (("adapter", "accepted_solution_semantics"), []),
                (("runtime", "python_version"), ""),
                (("records",), []),
                (("records", 0, "record_id"), " "),
                (("records", 0, "artifact_type"), "unknown_artifact"),
                (("records", 0, "evaluations"), True),
                (("records", 0, "evaluations"), 0),
                (("records", 0, "max_absolute_residual"), False),
            ],
        )

    def test_reserved_adapter_versions_scopes_and_configurations_are_enforced(self):
        self.assert_invalid_changes(
            1,
            [
                (("adapter", "version"), 2),
                (("adapter", "accepted_artifact_types"), ["callable_model"]),
                (("adapter", "configuration", "include_conditions"), 1),
                (("adapter", "configuration", "max_evaluations"), True),
                (("adapter", "configuration", "sampling"), "random"),
                (("adapter", "configuration", "decimal_precision"), True),
                (("adapter", "configuration", "points_per_axis"), False),
                (("adapter", "configuration", "tolerance"), True),
                (("adapter", "configuration", "extra"), None),
            ],
        )
        self.assert_invalid_changes(
            2,
            [
                (("adapter", "version"), 2),
                (("adapter", "configuration", "domain_checks"), 0),
                (("adapter", "configuration", "include_conditions"), 1),
                (("adapter", "configuration", "pipeline"), ["simplify"]),
                (("adapter", "configuration", "max_expression_ops"), True),
                (("adapter", "configuration", "symbolic_timeout_seconds"), False),
            ],
        )
        for version, other_version, policy in (
            (1, 2, "method_specific_obligation_diagnostics_no_global_verdict"),
            (2, 1, "method_specific_empirical_diagnostics_no_proof"),
        ):
            payload = _fixture(version)
            payload["baseline_report_version"] = other_version
            payload["evidence_policy"] = policy
            with self.subTest(version=version), self.assertRaises(AtlasBaselineError):
                validate_atlas_baseline_report(payload)

    def test_duplicate_record_and_symbolic_obligation_ids_are_rejected(self):
        payload = _fixture()
        payload["records"].append(copy.deepcopy(payload["records"][0]))
        with self.assertRaises(AtlasBaselineError):
            validate_atlas_baseline_report(payload)
        payload = _fixture(2)
        payload["records"][0]["checks"].append(copy.deepcopy(payload["records"][0]["checks"][0]))
        with self.assertRaises(AtlasBaselineError):
            validate_atlas_baseline_report(payload)

    def test_empirical_pass_and_failure_evidence_and_witness_must_agree(self):
        self.assert_invalid_changes(
            1,
            [
                (("records", 0, "evidence_kind"), "ABSTENTION"),
                (("records", 0, "evidence_level"), "EXACT"),
                (("records", 0, "max_absolute_residual"), "infinity"),
                (("records", 0, "reason"), ""),
                (("records", 0, "witness"), _fixture()["records"][1]["witness"]),
                (("records", 1, "evidence_kind"), "EMPIRICAL_PASS"),
                (("records", 1, "witness"), None),
                (("records", 1, "reason"), "explanation"),
                (("records", 1, "witness", "absolute_residual"), 1.0),
                (("records", 1, "witness", "constraint"), " "),
                (("records", 1, "witness", "sampled_inputs", "x"), True),
                (("records", 1, "witness", "sampled_inputs", ""), 0.0),
            ],
        )
        payload = _fixture()
        payload["records"][1]["max_absolute_residual"] = "infinity"
        payload["records"][1]["witness"]["absolute_residual"] = "infinity"
        self.assertEqual(validate_atlas_baseline_report(payload), payload)

    def test_fixed_collocation_outcomes_respect_recorded_tolerance_and_limit(self):
        self.assert_invalid_changes(
            1,
            [
                (("records", 0, "max_absolute_residual"), 1e-8),
                (("records", 0, "evaluations"), 1_000_001),
                (("records", 1, "evaluations"), 1_000_001),
            ],
        )
        for residual in (0.0, 1e-9):
            payload = _fixture()
            payload["records"][1]["max_absolute_residual"] = residual
            payload["records"][1]["witness"]["absolute_residual"] = residual
            with self.subTest(residual=residual), self.assertRaises(AtlasBaselineError):
                validate_atlas_baseline_report(payload)
        payload = _fixture()
        payload["records"][0]["max_absolute_residual"] = 1e-9
        self.assertEqual(validate_atlas_baseline_report(payload), payload)

    def test_unsupported_rows_require_only_abstention_with_a_reason(self):
        self.assert_invalid_changes(
            1,
            [
                (("records", 2, "evidence_kind"), "EMPIRICAL_PASS"),
                (("records", 2, "evidence_level"), "EMPIRICAL"),
                (("records", 2, "evaluations"), 1),
                (("records", 2, "max_absolute_residual"), 0.0),
                (("records", 2, "witness"), _fixture()["records"][1]["witness"]),
                (("records", 2, "reason"), None),
                (("records", 2, "reason"), " "),
            ],
        )

    def test_symbolic_checks_and_derived_outcomes_must_be_consistent(self):
        self.assert_invalid_changes(
            2,
            [
                (("records", 0, "outcome"), "nonzero"),
                (("records", 1, "outcome"), "undecided"),
                (("records", 2, "outcome"), "zero"),
                (("records", 0, "scope"), "global_proof"),
                (("records", 0, "checks"), []),
                (("records", 0, "checks", 0, "obligation_id"), "pde[0]"),
                (("records", 0, "checks", 0, "simplified_residual"), "1"),
                (("records", 0, "checks", 0, "materialized_residual"), None),
                (("records", 0, "checks", 0, "reason"), ""),
                (("records", 0, "checks", 0, "evidence_level"), "EMPIRICAL"),
                (("records", 1, "checks", 0, "simplified_residual"), "0"),
                (("records", 2, "checks", 0, "reason"), " "),
                (("records", 2, "checks", 0, "evidence_kind"), "CAS_RESIDUAL_ZERO"),
                (("records", 2, "checks", 0, "evidence_level"), "EXACT"),
            ],
        )

    def test_adapter_artifact_scope_and_direct_sympy_result_kind_are_enforced(self):
        self.assert_invalid_changes(1, [(("records", 0, "artifact_type"), "callable_model")])
        self.assert_invalid_changes(2, [(("records", 0, "artifact_type"), "callable_model")])
        for empirical_row in _fixture()["records"][:2]:
            payload = _fixture(2)
            payload["records"][0] = empirical_row
            with (
                self.subTest(outcome=empirical_row["outcome"]),
                self.assertRaises(AtlasBaselineError),
            ):
                validate_atlas_baseline_report(payload)

    def test_external_version_two_adapter_may_mix_empirical_and_symbolic_rows(self):
        payload = _fixture(2)
        payload["adapter"] = {
            "accepted_artifact_types": ["symbolic_expression", "callable_model"],
            "accepted_solution_semantics": ["classical_strong"],
            "configuration": {"synthetic_settings": {"flags": [True, None, "example"]}},
            "id": "synthetic_external_adapter",
            "version": 7,
        }
        payload["records"].append(_fixture()["records"][0])
        self.assertEqual(validate_atlas_baseline_report(payload), payload)
        payload["records"][0]["artifact_type"] = "callable_model"
        with self.assertRaises(AtlasBaselineError):
            validate_atlas_baseline_report(payload)

    def test_digest_is_canonical_defensive_and_preserves_array_order(self):
        payload = _fixture(2)
        payload["atlas"]["name"] = "Synthetic transport λ"
        before = copy.deepcopy(payload)
        normalized = validate_atlas_baseline_report(payload)
        canonical = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = atlas_baseline_report_sha256(payload)
        self.assertEqual(digest, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(
            digest, atlas_baseline_report_sha256(dict(reversed(list(payload.items()))))
        )
        reordered = copy.deepcopy(payload)
        reordered["records"].reverse()
        self.assertNotEqual(digest, atlas_baseline_report_sha256(reordered))
        self.assertEqual(payload, before)
        with self.assertRaises(AtlasBaselineError):
            atlas_baseline_report_sha256({})

    def test_large_integer_witness_is_preserved_and_not_hashed_as_a_float(self):
        payload = load_atlas_baseline_report(FIXTURES / "atlas-baseline-v1.json")
        point = payload["records"][1]["witness"]["sampled_inputs"]
        self.assertEqual(point["n"], 9007199254740993)
        self.assertIs(type(point["n"]), int)
        as_float = copy.deepcopy(payload)
        as_float["records"][1]["witness"]["sampled_inputs"]["n"] = float(point["n"])
        self.assertNotEqual(
            atlas_baseline_report_sha256(payload), atlas_baseline_report_sha256(as_float)
        )

    def test_non_json_values_and_nonstring_configuration_keys_are_rejected(self):
        for value in ({1: "coerced key"}, {"tuple": (1, 2)}, {"set": {1}}, {"bytes": b"x"}):
            payload = _fixture()
            payload["adapter"]["id"] = "synthetic_external_adapter"
            payload["adapter"]["configuration"] = value
            with self.subTest(value=value), self.assertRaises(AtlasBaselineError):
                validate_atlas_baseline_report(payload)
        for value in (None, [], "{}", True):
            with self.subTest(value=value), self.assertRaises(AtlasBaselineError):
                validate_atlas_baseline_report(value)
        for value in (float("nan"), float("inf"), -float("inf"), 10**400):
            payload = _fixture()
            payload["records"][1]["witness"]["sampled_inputs"]["x"] = value
            with self.subTest(value=repr(value)), self.assertRaises(AtlasBaselineError):
                validate_atlas_baseline_report(payload)

    def test_loader_rejects_duplicate_keys_and_nonfinite_json_numbers(self):
        cases = [
            '{"baseline_report_version": 1, "baseline_report_version": 1}',
            '{"adapter": {"configuration": {"x": 1, "x": 2}}}',
        ]
        rendered = json.dumps(_fixture())
        for number in ("NaN", "Infinity", "-Infinity", "1e400"):
            cases.append(rendered.replace('"x": 0.5', f'"x": {number}'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for document in cases:
                path.write_text(document, encoding="utf-8")
                with self.subTest(document=document[:80]), self.assertRaises(AtlasBaselineError):
                    load_atlas_baseline_report(path)

    def test_loader_rejects_invalid_json_utf8_and_excessive_nesting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for document in (b"{", b"\xff", ("[" * 1500 + "0" + "]" * 1500).encode()):
                path.write_bytes(document)
                with self.subTest(prefix=document[:20]), self.assertRaises(AtlasBaselineError):
                    load_atlas_baseline_report(path)

    def test_validator_wraps_excessive_configuration_nesting(self):
        nested = None
        for _ in range(1500):
            nested = [nested]
        payload = _fixture()
        payload["adapter"]["id"] = "synthetic_external_adapter"
        payload["adapter"]["configuration"] = {"nested": nested}
        with self.assertRaises(AtlasBaselineError):
            validate_atlas_baseline_report(payload)

    def test_loader_requires_a_regular_nonsymlink_file_within_size_limit(self):
        self.assertEqual(ATLAS_BASELINE_MAX_BYTES, 64 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report.json"
            path.write_text(json.dumps(_fixture()), encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(path)
            for invalid in (root, root / "missing.json", link):
                with self.subTest(path=invalid), self.assertRaises(AtlasBaselineError):
                    load_atlas_baseline_report(invalid)
            with patch("pdecert.atlas_baseline_reports.ATLAS_BASELINE_MAX_BYTES", 10):
                with self.assertRaises(AtlasBaselineError):
                    load_atlas_baseline_report(path)

    def test_cli_valid_reports_only_descriptive_internal_consistency_summary(self):
        for version in (1, 2):
            output = io.StringIO()
            with self.subTest(version=version), redirect_stdout(output):
                exit_code = main(
                    [
                        "corpus",
                        "validate-baseline",
                        str(FIXTURES / f"atlas-baseline-v{version}.json"),
                    ]
                )
            payload = _fixture(version)
            summary = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                set(summary),
                {
                    "adapter",
                    "atlas",
                    "baseline_report_version",
                    "outcomes",
                    "records",
                    "report_sha256",
                    "validation_scope",
                },
            )
            self.assertEqual(summary["validation_scope"], "report_internal_consistency_only")
            self.assertEqual(summary["report_sha256"], atlas_baseline_report_sha256(payload))
            self.assertEqual(summary["adapter"], payload["adapter"])
            self.assertEqual(summary["atlas"], payload["atlas"])
            self.assertEqual(summary["baseline_report_version"], version)
            self.assertEqual(summary["records"], len(payload["records"]))
            self.assertEqual(
                summary["outcomes"],
                {"pass": 1, "fail": 1, "unsupported": 1}
                if version == 1
                else {"zero": 1, "nonzero": 1, "undecided": 1, "unsupported": 1},
            )

    def test_cli_invalid_report_returns_input_error_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")
            for path in (invalid, Path(directory) / "missing.json"):
                output, errors = io.StringIO(), io.StringIO()
                with self.subTest(path=path), redirect_stdout(output), redirect_stderr(errors):
                    exit_code = main(["corpus", "validate-baseline", str(path)])
                self.assertEqual(exit_code, INPUT_ERROR)
                self.assertEqual(exit_code, 64)
                self.assertEqual(output.getvalue(), "")
                self.assertTrue(errors.getvalue().strip())
