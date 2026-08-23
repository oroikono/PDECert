import json
import tempfile
import unittest
from pathlib import Path

import sympy as sp

from experiments.adversarial_heat import build_cases
from pdecert import (
    SchemaError,
    Status,
    VerificationCase,
    case_from_dict,
    case_to_dict,
    dump_case,
    load_case,
    verify,
)


class JsonSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        heat_case = build_cases()[0]
        cls.case = VerificationCase(heat_case.problem, (heat_case.candidate,))
        cls.payload = case_to_dict(cls.case)

    def test_round_trip_preserves_a_proved_case(self):
        loaded = case_from_dict(self.payload)
        self.assertEqual(loaded.problem.name, self.case.problem.name)
        self.assertEqual([str(item) for item in loaded.problem.variables], ["x", "t"])
        self.assertEqual(verify(loaded.problem, loaded.candidate_expressions).status, Status.PROVED)
        self.assertEqual(
            sp.simplify(loaded.candidate_expressions[0] - self.case.candidate_expressions[0]), 0
        )

    def test_version_one_input_remains_readable(self):
        payload = dict(self.payload)
        payload["schema_version"] = 1
        del payload["parameters"]
        loaded = case_from_dict(payload)
        self.assertEqual(loaded.problem.parameter_assumptions, {})
        self.assertEqual(case_to_dict(loaded)["schema_version"], 2)

    def test_parameter_assumptions_round_trip_into_sympy_symbols(self):
        experiment = build_cases()[5]
        case = VerificationCase(experiment.problem, (experiment.candidate,))
        payload = case_to_dict(case)
        self.assertEqual(payload["parameters"], {"k": ["positive"]})

        loaded = case_from_dict(payload)
        parameter = loaded.problem.variables[2]
        self.assertTrue(parameter.is_positive)
        self.assertEqual(
            loaded.problem.parameter_assumptions[parameter],
            frozenset({"positive"}),
        )
        report = verify(loaded.problem, loaded.candidate_expressions)
        self.assertEqual(report.status, Status.REFUTED)
        self.assertEqual(report.witness.point["k"], 0.2)

    def test_file_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            dump_case(self.case, path)
            loaded = load_case(path)
            self.assertEqual(case_to_dict(loaded), self.payload)
            self.assertTrue(path.read_text().endswith("\n"))

    def test_missing_field_has_a_path_in_the_error(self):
        payload = dict(self.payload)
        del payload["schema_version"]
        with self.assertRaisesRegex(SchemaError, r"^\$: missing field\(s\): schema_version$"):
            case_from_dict(payload)

    def test_domains_must_match_declared_variables(self):
        payload = dict(self.payload)
        payload["domains"] = {"x": [0, 1], "y": [0, 1]}
        with self.assertRaisesRegex(SchemaError, r"\$\.domains: missing: t; unknown: y"):
            case_from_dict(payload)

    def test_unknown_symbol_is_rejected(self):
        payload = dict(self.payload)
        payload["candidate_expressions"] = ["sin(pi*y)"]
        with self.assertRaisesRegex(SchemaError, "unknown symbol: y"):
            case_from_dict(payload)

    def test_unknown_parameter_is_rejected(self):
        payload = dict(self.payload)
        payload["parameters"] = {"k": ["positive"]}
        with self.assertRaisesRegex(SchemaError, r"unknown parameter\(s\): k"):
            case_from_dict(payload)

    def test_unsupported_parameter_assumption_is_rejected(self):
        payload = dict(self.payload)
        payload["parameters"] = {"x": ["prime"]}
        with self.assertRaisesRegex(SchemaError, "unsupported assumption: prime"):
            case_from_dict(payload)

    def test_conflicting_parameter_assumptions_are_rejected(self):
        payload = dict(self.payload)
        payload["parameters"] = {"x": ["positive", "negative"]}
        with self.assertRaisesRegex(SchemaError, "assumptions are inconsistent"):
            case_from_dict(payload)

    def test_parameter_assumption_must_match_domain(self):
        payload = dict(self.payload)
        payload["parameters"] = {"x": ["positive"]}
        with self.assertRaisesRegex(SchemaError, "domain for positive parameter x"):
            case_from_dict(payload)

    def test_non_increasing_domain_is_rejected(self):
        payload = dict(self.payload)
        payload["domains"] = {"x": [1, 0], "t": [0, 1]}
        with self.assertRaisesRegex(SchemaError, "bounds must be finite and increasing"):
            case_from_dict(payload)

    def test_expression_does_not_allow_attribute_access(self):
        payload = dict(self.payload)
        payload["candidate_expressions"] = ["__import__('os').system('echo unsafe')"]
        with self.assertRaisesRegex(SchemaError, "unsupported expression syntax"):
            case_from_dict(payload)

    def test_unknown_fields_are_rejected(self):
        payload = dict(self.payload)
        payload["typo"] = True
        with self.assertRaisesRegex(SchemaError, r"unknown field\(s\): typo"):
            case_from_dict(payload)

    def test_duplicate_constraint_names_are_rejected(self):
        payload = dict(self.payload)
        payload["conditions"] = [
            {"name": "heat PDE", "expression": "0"},
        ]
        with self.assertRaisesRegex(SchemaError, "duplicate constraint name: heat PDE"):
            case_from_dict(payload)

    def test_serializer_rejects_an_undeclared_candidate_symbol(self):
        y = sp.Symbol("y", real=True)
        invalid = VerificationCase(self.case.problem, (y,))
        with self.assertRaisesRegex(SchemaError, "unknown symbol: y"):
            case_to_dict(invalid)

    def test_invalid_json_is_reported_without_a_traceback_from_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text("{")
            with self.assertRaisesRegex(SchemaError, "invalid JSON"):
                load_case(path)

    def test_serialized_payload_is_plain_json(self):
        json.dumps(self.payload)


if __name__ == "__main__":
    unittest.main()
