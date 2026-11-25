import builtins
import copy
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from pdecert import SolutionArtifact, verify_artifact
from pdecert import gridded
from pdecert.core import Constraint, Problem
from pdecert.gridded import (
    GRID_ARTIFACT_MAX_BYTES,
    GRID_ARTIFACT_MAX_VALUES,
    GRID_ARTIFACT_VERSION,
    GridArtifactError,
    GridCandidate,
    grid_from_dict,
    grid_to_dict,
    load_grid,
    write_grid,
)


def minimal_grid():
    """An asymmetric grid with a recognizable last-axis-fastest field."""

    return {
        "schema_version": 1,
        "artifact_kind": "rectilinear_grid",
        "artifact_id": "asymmetric-grid",
        "problem_id": "test-problem",
        "axes": [
            {"name": "x", "coordinates": [-1.0, 2.0]},
            {"name": "t", "coordinates": [0.0, 0.25, 1.0]},
        ],
        "fields": [
            {"name": "u", "values": [-10.0, -9.75, -9.0, 20.0, 20.25, 21.0]},
            {"name": "v", "values": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]},
        ],
        "discretization": "Stored nodal samples on the declared Cartesian product.",
        "provenance": {"source": "unit-test", "note": "München"},
        "value_location": "nodes",
        "storage_order": "last_axis_fastest",
        "dtype": "float64",
    }


def candidate_from_payload(payload):
    return GridCandidate(
        artifact_id=payload["artifact_id"],
        problem_id=payload["problem_id"],
        axes=[(axis["name"], axis["coordinates"]) for axis in payload["axes"]],
        fields=[(field["name"], field["values"]) for field in payload["fields"]],
        discretization=payload["discretization"],
        provenance=payload["provenance"],
    )


class GridArtifactTests(unittest.TestCase):
    def test_committed_example_and_serialization_match_the_public_schema(self):
        schema = json.loads(Path("schema/grid-artifact-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        example = Path("examples/rectilinear-grid.json")
        payload = json.loads(example.read_text())
        validator.validate(payload)
        grid = load_grid(example)
        validator.validate(grid_to_dict(grid))
        self.assertIsInstance(grid, SolutionArtifact)
        payload["axes"][0]["name"] = "x\n"
        self.assertFalse(validator.is_valid(payload))

    def test_reference_example_preserves_node_order_and_empirical_scope(self):
        from examples.gridded_reference_comparison import main

        report = main()
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["evidence_level"], "EMPIRICAL")
        self.assertNotIn("status", report)
        field = report["fields"][0]
        self.assertEqual(field["max_absolute_error"], 0.125)
        self.assertAlmostEqual(field["rmse"], 0.125 / math.sqrt(6))
        self.assertEqual(field["max_error_sample"]["index"], 3)
        self.assertEqual(field["max_error_sample"]["coordinates"], {"x": 0.25, "t": 1.0})

    def test_public_limits_and_round_trip_preserve_asymmetric_layout(self):
        self.assertEqual(GRID_ARTIFACT_VERSION, 1)
        self.assertEqual(GRID_ARTIFACT_MAX_VALUES, 200_000)
        self.assertEqual(GRID_ARTIFACT_MAX_BYTES, 4_000_000)
        self.assertTrue(issubclass(GridArtifactError, ValueError))
        payload = minimal_grid()
        artifact = grid_from_dict(payload)
        self.assertIsInstance(artifact, GridCandidate)
        self.assertEqual(artifact.kind, "rectilinear_grid")
        self.assertEqual(artifact.shape, (2, 3))
        self.assertEqual(artifact.coordinate_names, ("x", "t"))
        self.assertEqual(artifact.field_names, ("u", "v"))
        self.assertEqual(artifact.axes, (("x", (-1.0, 2.0)), ("t", (0.0, 0.25, 1.0))))
        self.assertEqual(artifact.fields[0], ("u", (-10.0, -9.75, -9.0, 20.0, 20.25, 21.0)))
        for i, x in enumerate(artifact.axes[0][1]):
            for j, t in enumerate(artifact.axes[1][1]):
                self.assertEqual(artifact.fields[0][1][i * 3 + j], 10 * x + t)
        self.assertEqual(grid_to_dict(artifact), payload)
        self.assertEqual(grid_from_dict(grid_to_dict(artifact)), artifact)
        self.assertEqual(candidate_from_payload(payload), artifact)

    def test_candidate_and_serialized_values_are_deeply_independent(self):
        for constructor in (grid_from_dict, candidate_from_payload):
            with self.subTest(constructor=constructor.__name__):
                payload = minimal_grid()
                expected = copy.deepcopy(payload)
                artifact = constructor(payload)
                payload["axes"][0]["coordinates"][0] = -100.0
                payload["fields"][0]["values"][0] = 100.0
                payload["provenance"]["source"] = "changed"
                self.assertEqual(grid_to_dict(artifact), expected)
                self.assertIsInstance(artifact.axes, tuple)
                self.assertIsInstance(artifact.fields, tuple)
                for name, values in (*artifact.axes, *artifact.fields):
                    self.assertIsInstance(name, str)
                    self.assertIsInstance(values, tuple)
                    self.assertTrue(all(type(value) is float for value in values))
                with self.assertRaises(FrozenInstanceError):
                    artifact.artifact_id = "changed"
                with self.assertRaises(TypeError):
                    artifact.fields[0][1][0] = 1.0
                with self.assertRaises(TypeError):
                    artifact.provenance["source"] = "changed"
                serialized = grid_to_dict(artifact)
                serialized["axes"][0]["coordinates"][0] = -100.0
                serialized["fields"][0]["values"][0] = 100.0
                serialized["provenance"]["source"] = "changed"
                self.assertEqual(grid_to_dict(artifact), expected)

    def test_singleton_axes_and_separate_name_namespaces_are_allowed(self):
        payload = minimal_grid()
        payload["axes"] = [{"name": "u", "coordinates": [2]}]
        payload["fields"] = [{"name": "u", "values": [2**54]}]
        artifact = grid_from_dict(payload)
        self.assertEqual(artifact.shape, (1,))
        self.assertEqual(artifact.axes, (("u", (2.0,)),))
        self.assertEqual(artifact.fields, (("u", (float(2**54),)),))

    def test_required_keys_and_unknown_keys_are_rejected_at_every_object(self):
        baseline = minimal_grid()
        for key in baseline:
            with self.subTest(missing=key):
                payload = copy.deepcopy(baseline)
                del payload[key]
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)
        for path in ((), ("axes", 0), ("fields", 0)):
            with self.subTest(extra_at=path):
                payload = minimal_grid()
                target = payload
                for key in path:
                    target = target[key]
                target["unexpected"] = "rejected"
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)
        for container, key in (
            ("axes", "name"),
            ("axes", "coordinates"),
            ("fields", "name"),
            ("fields", "values"),
        ):
            with self.subTest(container=container, missing=key):
                payload = minimal_grid()
                del payload[container][0][key]
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)

    def test_unsupported_format_markers_and_noninteger_versions_are_rejected(self):
        invalid = {
            "schema_version": [0, 2, True, 1.0, "1", None],
            "artifact_kind": ["grid", "symbolic", None],
            "value_location": ["cells", None],
            "storage_order": ["first_axis_fastest", None],
            "dtype": ["float32", "int64", None],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    payload = minimal_grid()
                    payload[key] = value
                    with self.assertRaises(GridArtifactError):
                        grid_from_dict(payload)

    def test_ids_discretization_and_provenance_require_meaningful_strings(self):
        for key in ("artifact_id", "problem_id", "discretization"):
            for value in ("", " \n\t", None, 1):
                with self.subTest(key=key, value=value):
                    payload = minimal_grid()
                    payload[key] = value
                    with self.assertRaises(GridArtifactError):
                        grid_from_dict(payload)
        for provenance in (
            {},
            [],
            {"source": ""},
            {"source": " "},
            {"": "source"},
            {" ": "source"},
            {1: "source"},
            {"source": 1},
            {"source": {"nested": "value"}},
        ):
            with self.subTest(provenance=provenance):
                payload = minimal_grid()
                payload["provenance"] = provenance
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)

    def test_shape_containers_names_and_monotonicity_are_validated(self):
        for container in ("axes", "fields"):
            for value in ([], {}, "x", None, [None]):
                with self.subTest(container=container, value=value):
                    payload = minimal_grid()
                    payload[container] = value
                    with self.assertRaises(GridArtifactError):
                        grid_from_dict(payload)
            for name in ("", "x-y", "1x", " x", "é", "x.y", None, 1):
                with self.subTest(container=container, name=name):
                    payload = minimal_grid()
                    payload[container][0]["name"] = name
                    with self.assertRaises(GridArtifactError):
                        grid_from_dict(payload)
            payload = minimal_grid()
            payload[container][1]["name"] = payload[container][0]["name"]
            with self.subTest(duplicate=container), self.assertRaises(GridArtifactError):
                grid_from_dict(payload)
        for coordinates in ([], [1.0, 1.0], [1.0, 0.0], [-0.0, 0.0], [[0.0], [1.0]], "01", None):
            with self.subTest(coordinates=coordinates):
                payload = minimal_grid()
                payload["axes"][0]["coordinates"] = coordinates
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)
        for values in ([], [1.0] * 5, [1.0] * 7, [[1.0, 2.0, 3.0]] * 2, "123456", None):
            with self.subTest(field_values=values):
                payload = minimal_grid()
                payload["fields"][0]["values"] = values
                with self.assertRaises(GridArtifactError):
                    grid_from_dict(payload)

    def test_scalar_validation_applies_to_axes_fields_and_direct_construction(self):
        invalid = (
            True,
            False,
            None,
            "1.0",
            complex(1, 0),
            math.nan,
            math.inf,
            -math.inf,
            2**53 + 1,
            10**400,
        )
        for container, values_key in (("axes", "coordinates"), ("fields", "values")):
            for scalar in invalid:
                with self.subTest(container=container, scalar=scalar):
                    payload = minimal_grid()
                    payload[container][0][values_key][0] = scalar
                    for constructor in (grid_from_dict, candidate_from_payload):
                        with self.assertRaises(GridArtifactError):
                            constructor(payload)

    def test_negative_zero_is_canonicalized_in_memory_and_serialization(self):
        payload = minimal_grid()
        payload["axes"][1]["coordinates"][0] = -0.0
        payload["fields"][0]["values"][0] = -0.0
        artifact = grid_from_dict(payload)
        serialized = grid_to_dict(artifact)
        for value in (
            artifact.axes[1][1][0],
            artifact.fields[0][1][0],
            serialized["axes"][1]["coordinates"][0],
            serialized["fields"][0]["values"][0],
        ):
            self.assertEqual(value, 0.0)
            self.assertEqual(math.copysign(1.0, value), 1.0)

    def test_value_budget_counts_all_axes_and_fields_at_the_exact_boundary(self):
        payload = minimal_grid()
        count = 2 + 3 + 6 + 6
        with mock.patch.object(gridded, "GRID_ARTIFACT_MAX_VALUES", count):
            self.assertEqual(grid_from_dict(payload).shape, (2, 3))
        with mock.patch.object(gridded, "GRID_ARTIFACT_MAX_VALUES", count - 1):
            for constructor in (grid_from_dict, candidate_from_payload):
                with self.assertRaises(GridArtifactError):
                    constructor(payload)

    def test_writing_is_deterministic_utf8_and_does_not_replace_existing_files(self):
        artifact = grid_from_dict(minimal_grid())
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_grid(artifact, first)
            write_grid(artifact, second)
            rendered = first.read_bytes()
            self.assertEqual(rendered, second.read_bytes())
            self.assertEqual(json.loads(rendered.decode("utf-8")), grid_to_dict(artifact))
            self.assertEqual(load_grid(first), artifact)
            with self.assertRaises(FileExistsError):
                write_grid(artifact, first)
            self.assertEqual(first.read_bytes(), rendered)
            sentinel = Path(directory) / "existing.json"
            sentinel.write_bytes(b"existing evidence\n")
            with self.assertRaises(FileExistsError):
                write_grid(artifact, sentinel)
            self.assertEqual(sentinel.read_bytes(), b"existing evidence\n")

    def test_strict_json_rejects_duplicates_nonfinite_invalid_utf8_and_deep_nesting(self):
        serialized = json.dumps(minimal_grid())
        malformed = [
            b"\xff",
            b"{",
            b"[]",
            b"null",
            (serialized + " {}").encode(),
            serialized.replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ).encode(),
            serialized.replace('"name": "x"', '"name": "x", "name": "x"').encode(),
            serialized.replace(
                '"source": "unit-test"', '"source": "unit-test", "source": "unit-test"'
            ).encode(),
            ("[" * 1200 + "0" + "]" * 1200).encode(),
        ]
        numeric_payload = minimal_grid()
        numeric_payload["fields"][0]["values"][0] = 314159.0
        numeric_json = json.dumps(numeric_payload)
        for token in (
            "NaN",
            "Infinity",
            "-Infinity",
            "1e9999",
            "-1e9999",
            "1e-9999",
            "-1e-9999",
            "9007199254740993",
        ):
            malformed.append(numeric_json.replace("314159.0", token).encode())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for index, raw in enumerate(malformed):
                with self.subTest(case=index):
                    path.write_bytes(raw)
                    with self.assertRaises(GridArtifactError):
                        load_grid(path)

    def test_byte_budget_is_measured_in_bytes_and_accepts_the_exact_limit(self):
        raw = json.dumps(minimal_grid(), ensure_ascii=False).encode("utf-8")
        self.assertGreater(len(raw), len(raw.decode("utf-8")))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            path.write_bytes(raw)
            with mock.patch.object(gridded, "GRID_ARTIFACT_MAX_BYTES", len(raw)):
                self.assertEqual(load_grid(path).shape, (2, 3))
                path.write_bytes(raw + b" ")
                with self.assertRaises(GridArtifactError):
                    load_grid(path)

    def test_writer_rejects_oversized_metadata_before_creating_a_file(self):
        artifact = grid_from_dict(minimal_grid())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            with mock.patch.object(gridded, "GRID_ARTIFACT_MAX_BYTES", 10):
                with self.assertRaises(GridArtifactError):
                    write_grid(artifact, path)
            self.assertFalse(path.exists())

    def test_grid_operations_do_not_import_numpy_or_torch(self):
        original_import = builtins.__import__

        def reject_optional_numerics(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch"}:
                raise AssertionError(f"grid representation imported {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_optional_numerics):
            artifact = grid_from_dict(minimal_grid())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "grid.json"
                write_grid(artifact, path)
                self.assertEqual(load_grid(path), artifact)

    def test_existing_verifier_rejects_grid_without_a_backend(self):
        import sympy as sp

        x = sp.Symbol("x", real=True)
        problem = Problem(
            name="test-problem",
            variables=(x,),
            domains={x: (-1.0, 2.0)},
            pde_residuals=(Constraint("constant", sp.Integer(0)),),
        )
        with self.assertRaisesRegex(TypeError, "unsupported problem/artifact pair"):
            verify_artifact(problem, grid_from_dict(minimal_grid()))


if __name__ == "__main__":
    unittest.main()
