import copy
import hashlib
import json
import math
import random
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from pdecert import (
    REFERENCE_COMPARISON_MAX_VALUES,
    REFERENCE_COMPARISON_VERSION,
    ReferenceComparisonError,
    compare_reference_fields,
)


def sample_inputs():
    return {
        "coordinates": {"x": [0, 1, 2]},
        "candidate": {"u": [1, 2, 3]},
        "reference": {"u": [1, 2, 3]},
        "problem_id": "supplied-problem",
        "candidate_id": "supplied-candidate",
        "reference_id": "supplied-reference",
        "reference_description": "Values supplied by the caller",
        "reference_uncertainty": "No validated uncertainty estimate is available",
        "sampling_description": "Three equally weighted supplied rows",
    }


def compare(**changes):
    inputs = sample_inputs()
    inputs.update(changes)
    return compare_reference_fields(**inputs)


def single_field(predictions, references):
    return compare(
        coordinates={"x": list(range(len(predictions)))},
        candidate={"u": predictions},
        reference={"u": references},
    )["fields"][0]


def test_exact_values_remain_empirical_without_a_pde_verdict():
    report = compare()
    field = report["fields"][0]
    assert report["reference_comparison_version"] == REFERENCE_COMPARISON_VERSION == 1
    assert REFERENCE_COMPARISON_MAX_VALUES == 1_000_000
    assert report["evidence_level"] == "EMPIRICAL"
    assert report["scope"] == "supplied_field_values_at_supplied_points_only"
    assert report["sample_count"] == 3
    assert field["rmse"] == field["relative_l2"] == field["max_absolute_error"] == 0
    assert field["rmse_unavailable_reason"] is None
    assert field["relative_l2_unavailable_reason"] is None
    assert field["max_error_sample"] == {
        "index": 0,
        "coordinates": {"x": 0.0},
        "candidate_value": 1.0,
        "reference_value": 1.0,
    }
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized) == report
    for unsupported_claim in (
        "status",
        "outcome",
        "certificate",
        "solution_error_bound",
        "witness",
        "PROVED",
        "REFUTED",
    ):
        assert unsupported_claim not in serialized


def test_shifted_field_has_standard_discrete_metrics():
    field = single_field([2, 3, 4], [1, 2, 3])
    assert field["rmse"] == pytest.approx(1)
    assert field["relative_l2"] == pytest.approx(math.sqrt(3 / 14))
    assert field["max_absolute_error"] == 1


def test_fields_are_separate_and_sorted_without_cross_unit_aggregation():
    report = compare(
        coordinates={"t": [0, 1, 2], "x": [3, 4, 5]},
        candidate={"v": [12, 22, 32], "u": [1, 2, 3]},
        reference={"u": [1, 2, 3], "v": [10, 20, 30]},
    )
    assert report["coordinate_names"] == ["t", "x"]
    assert [row["field"] for row in report["fields"]] == ["u", "v"]
    assert [row["rmse"] for row in report["fields"]] == [0, 2]
    assert "rmse" not in report
    assert "relative_l2" not in report


def test_first_tied_maximum_identifies_the_original_row_and_values():
    report = compare(
        coordinates={"x": [10, -10, 20], "t": [2, 1, 3]},
        candidate={"u": [1, 5, -1]},
        reference={"u": [1, 2, 2]},
    )
    field = report["fields"][0]
    assert field["rmse"] == pytest.approx(math.sqrt(6))
    assert field["max_error_sample"] == {
        "index": 1,
        "coordinates": {"t": 1.0, "x": -10.0},
        "candidate_value": 5.0,
        "reference_value": 2.0,
    }


def test_duplicate_coordinates_retain_equal_weight_per_row():
    report = compare(
        coordinates={"x": [0, 0, 1]},
        candidate={"u": [1, 1, 0]},
        reference={"u": [0, 0, 0]},
    )
    assert report["sample_count"] == 3
    assert report["fields"][0]["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert report["configuration"]["reduction"] == "equal_weight_per_row_duplicates_retained"


def test_tuple_columns_and_non_template_problem_ids_are_supported():
    report = compare(
        coordinates={"coordinate in caller units": (0, 1, 2)},
        candidate={"u": (1, 2, 3)},
        reference={"u": (1, 2, 3)},
        problem_id="not-a-registered-template",
    )
    assert report["problem_id"] == "not-a-registered-template"
    assert report["coordinate_names"] == ["coordinate in caller units"]


def test_declared_reference_quality_is_preserved_not_verified():
    report = compare(reference_uncertainty="  caller's unvalidated estimate  ")
    assert report["reference"] == {
        "id": "supplied-reference",
        "description": "Values supplied by the caller",
        "uncertainty": "  caller's unvalidated estimate  ",
        "assessment": "caller_supplied_not_verified",
    }
    assert report["configuration"]["units"] == "as_supplied_no_conversion"
    assert report["configuration"]["relative_norm"] == "per_field_discrete_l2"
    assert report["configuration"]["arithmetic"] == "python_binary64"
    assert report["runtime"]["python_version"]
    assert report["runtime"]["pdecert_version"]


def test_normalized_input_identity_includes_samples_and_metadata_only():
    inputs = sample_inputs()
    report = compare_reference_fields(**inputs)
    payload = {
        "metadata": {
            "problem_id": inputs["problem_id"],
            "candidate_id": inputs["candidate_id"],
            "reference": report["reference"],
            "sampling_description": inputs["sampling_description"],
        },
        "samples": {
            category: {
                name: [float(value) for value in values]
                for name, values in inputs[category].items()
            }
            for category in ("coordinates", "candidate", "reference")
        },
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert report["input_sha256"] == expected
    assert (
        report["input_identity_scope"] == "normalized_binary64_samples_and_declared_metadata_only"
    )


@pytest.mark.parametrize(
    "metadata_key",
    [
        "problem_id",
        "candidate_id",
        "reference_id",
        "reference_description",
        "reference_uncertainty",
        "sampling_description",
    ],
)
def test_changing_declared_metadata_changes_identity(metadata_key):
    inputs = sample_inputs()
    previous = compare_reference_fields(**inputs)["input_sha256"]
    inputs[metadata_key] += " changed"
    assert compare_reference_fields(**inputs)["input_sha256"] != previous


@pytest.mark.parametrize("category", ["coordinates", "candidate", "reference"])
def test_changing_any_supplied_column_changes_identity(category):
    inputs = sample_inputs()
    previous = compare_reference_fields(**inputs)["input_sha256"]
    next(iter(inputs[category].values()))[1] += 0.25
    assert compare_reference_fields(**inputs)["input_sha256"] != previous


def test_mapping_insertion_order_does_not_change_report_or_identity():
    inputs = sample_inputs()
    inputs["coordinates"] = {"z": [2, 3, 4], "x": [0, 1, 2]}
    inputs["candidate"] = {"v": [5, 6, 7], "u": [1, 2, 3]}
    inputs["reference"] = {"u": [1, 2, 3], "v": [5, 6, 7]}
    first = compare_reference_fields(**inputs)
    for category in ("coordinates", "candidate", "reference"):
        inputs[category] = dict(reversed(list(inputs[category].items())))
    assert compare_reference_fields(**inputs) == first


def test_row_order_is_preserved_in_identity_even_if_metrics_are_unchanged():
    inputs = sample_inputs()
    first = compare_reference_fields(**inputs)
    for category in ("coordinates", "candidate", "reference"):
        for values in inputs[category].values():
            values.reverse()
    second = compare_reference_fields(**inputs)
    assert first["input_sha256"] != second["input_sha256"]
    assert first["fields"][0]["rmse"] == second["fields"][0]["rmse"]


def test_exact_integer_float_and_signed_zero_inputs_have_one_normalized_identity():
    first = compare()
    second = compare(
        coordinates={"x": [-0.0, 1.0, 2.0]},
        candidate={"u": [1.0, 2.0, 3.0]},
        reference={"u": [1.0, 2.0, 3.0]},
    )
    assert first == second
    assert math.copysign(1, second["fields"][0]["max_error_sample"]["coordinates"]["x"]) == 1


def test_inputs_are_not_modified_or_retained_as_mutable_report_references():
    inputs = sample_inputs()
    original = copy.deepcopy(inputs)
    report = compare_reference_fields(**inputs)
    original_report = copy.deepcopy(report)
    assert inputs == original
    for category in ("coordinates", "candidate", "reference"):
        for values in inputs[category].values():
            values[0] = 99
    assert report == original_report


@pytest.mark.parametrize(
    "metadata_key",
    [
        "problem_id",
        "candidate_id",
        "reference_id",
        "reference_description",
        "reference_uncertainty",
        "sampling_description",
    ],
)
@pytest.mark.parametrize("bad_value", ["", " \t\n", None, 0, True, []])
def test_metadata_must_be_nonblank_text(metadata_key, bad_value):
    with pytest.raises(ReferenceComparisonError):
        compare(**{metadata_key: bad_value})


@pytest.mark.parametrize(
    "metadata_key",
    [
        "problem_id",
        "candidate_id",
        "reference_id",
        "reference_description",
        "reference_uncertainty",
        "sampling_description",
    ],
)
def test_provenance_metadata_cannot_be_omitted(metadata_key):
    inputs = sample_inputs()
    del inputs[metadata_key]
    with pytest.raises(TypeError):
        compare_reference_fields(**inputs)


@pytest.mark.parametrize("category", ["coordinates", "candidate", "reference"])
@pytest.mark.parametrize("bad_value", [{}, [], None, "u"])
def test_all_column_groups_must_be_nonempty_mappings(category, bad_value):
    with pytest.raises(ReferenceComparisonError):
        compare(**{category: bad_value})


@pytest.mark.parametrize("category", ["coordinates", "candidate", "reference"])
@pytest.mark.parametrize("name", ["", " \n", 1, None])
def test_column_names_must_be_nonblank_text(category, name):
    with pytest.raises(ReferenceComparisonError):
        compare(**{category: {name: [1, 2, 3]}})


@pytest.mark.parametrize(
    "bad_column",
    [
        [],
        [1, 2],
        [1, 2, 3, 4],
        "123",
        b"123",
        123,
        None,
        {0: 1, 1: 2, 2: 3},
        [True, 2, 3],
        [1j, 2, 3],
        ["1", 2, 3],
        [None, 2, 3],
        [math.nan, 2, 3],
        [math.inf, 2, 3],
        [-math.inf, 2, 3],
        [[1], [2], [3]],
        [2**53 + 1, 2, 3],
        [10**400, 2, 3],
    ],
)
@pytest.mark.parametrize("category", ["coordinates", "candidate", "reference"])
def test_invalid_or_unsupported_columns_are_rejected_without_broadcasting(category, bad_column):
    name = "x" if category == "coordinates" else "u"
    with pytest.raises(ReferenceComparisonError):
        compare(**{category: {name: bad_column}})


def test_a_generator_is_not_executed_as_a_sample_column():
    def forbidden_generator():
        raise AssertionError("A generator must not be executed")
        yield 0

    with pytest.raises(ReferenceComparisonError):
        compare(candidate={"u": forbidden_generator()})


def test_array_like_objects_require_explicit_caller_conversion():
    class ArrayLike:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

        def tolist(self):
            raise AssertionError("The metric must not implicitly convert framework arrays")

    with pytest.raises(ReferenceComparisonError):
        compare(candidate={"u": ArrayLike()})


@pytest.mark.parametrize(
    "candidate,reference",
    [
        ({"v": [1, 2, 3]}, {"u": [1, 2, 3]}),
        ({"u": [1, 2, 3], "v": [1, 2, 3]}, {"u": [1, 2, 3]}),
        ({"u": [1, 2, 3]}, {"u": [1, 2, 3], "v": [1, 2, 3]}),
    ],
)
def test_candidate_and_reference_field_sets_must_match(candidate, reference):
    with pytest.raises(ReferenceComparisonError):
        compare(candidate=candidate, reference=reference)


def test_all_coordinate_columns_must_have_the_same_length():
    with pytest.raises(ReferenceComparisonError):
        compare(coordinates={"x": [0, 1, 2], "t": [0, 1]})


def test_sample_budget_is_checked_before_reading_numeric_values():
    class OversizedColumn(Sequence):
        def __len__(self):
            return REFERENCE_COMPARISON_MAX_VALUES // 3 + 1

        def __getitem__(self, index):
            raise AssertionError("Oversized inputs must be rejected before value access")

    column = OversizedColumn()
    with pytest.raises(ReferenceComparisonError):
        compare(coordinates={"x": column}, candidate={"u": column}, reference={"u": column})


def test_large_exactly_representable_integer_is_allowed():
    field = single_field([2**100], [2**100])
    assert field["rmse"] == 0
    assert field["max_error_sample"]["reference_value"] == float(2**100)


@pytest.mark.parametrize("candidate", [[0, 0, 0], [1, 2, 3]])
def test_zero_reference_norm_has_no_fabricated_relative_error(candidate):
    field = single_field(candidate, [0, 0, 0])
    assert field["relative_l2"] is None
    assert field["relative_l2_unavailable_reason"] == "zero_reference_norm"
    assert field["rmse"] == pytest.approx(math.sqrt(sum(value**2 for value in candidate) / 3))


def test_scaled_norms_do_not_underflow_by_squaring_small_normal_values():
    field = single_field([2e-300, 2e-300], [1e-300, 1e-300])
    assert field["rmse"] == pytest.approx(1e-300, rel=1e-15, abs=0)
    assert field["relative_l2"] == pytest.approx(1)


def test_scaled_norms_do_not_overflow_by_squaring_large_values():
    field = single_field([1e308, 1e308], [0, 0])
    assert field["rmse"] == 1e308
    assert field["max_absolute_error"] == 1e308


def test_relative_norm_avoids_overflow_in_intermediate_scale_ratio():
    field = single_field([1e308, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    assert field["rmse"] == 5e307
    assert field["relative_l2"] == 1e308


def test_relative_norm_preserves_representable_subnormal_after_reduction():
    tiny = math.ulp(0.0)
    field = single_field([2, tiny, tiny, tiny, tiny], [2, 0, 0, 0, 0])
    assert field["relative_l2"] == tiny
    assert field["relative_l2_unavailable_reason"] is None


def test_unrepresentable_absolute_difference_rejects_the_comparison():
    with pytest.raises(ReferenceComparisonError):
        single_field([1e308], [-1e308])


def test_relative_norm_overflow_is_reported_as_unavailable_not_infinity():
    field = single_field([1], [math.ulp(0.0)])
    assert field["relative_l2"] is None
    assert field["relative_l2_unavailable_reason"] == "binary64_overflow"
    assert math.isfinite(field["rmse"])
    json.dumps(field, allow_nan=False)


def test_relative_norm_underflow_is_not_reported_as_exact_agreement():
    tiny = math.ulp(0.0)
    field = single_field([2, tiny], [2, 0])
    assert field["relative_l2"] is None
    assert field["relative_l2_unavailable_reason"] == "binary64_underflow"
    assert field["max_absolute_error"] == tiny


def test_rmse_underflow_is_not_reported_as_exact_agreement():
    tiny = math.ulp(0.0)
    field = single_field([tiny, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    assert field["rmse"] is None
    assert field["rmse_unavailable_reason"] == "binary64_underflow"
    assert field["max_absolute_error"] == tiny
    json.dumps(field, allow_nan=False)


def test_seeded_numeric_samples_agree_with_high_precision_discrete_formulas():
    rng = random.Random(20260908)
    with localcontext() as context:
        context.prec = 120
        for _ in range(64):
            count = rng.randint(1, 7)
            predictions = [rng.uniform(-2, 2) * 10 ** rng.randint(-120, 120) for _ in range(count)]
            references = [rng.uniform(-2, 2) * 10 ** rng.randint(-120, 120) for _ in range(count)]
            squared_errors = sum(
                (Decimal.from_float(prediction) - Decimal.from_float(reference)) ** 2
                for prediction, reference in zip(predictions, references)
            )
            squared_reference = sum(Decimal.from_float(value) ** 2 for value in references)
            expected_rmse = float((squared_errors / count).sqrt())
            expected_relative = float((squared_errors / squared_reference).sqrt())
            field = single_field(predictions, references)
            assert field["rmse"] == pytest.approx(expected_rmse, rel=1e-14, abs=0)
            assert field["relative_l2"] == pytest.approx(expected_relative, rel=1e-14, abs=0)


def comparison_validator():
    path = Path(__file__).resolve().parents[1] / "schema" / "reference-comparison-v1.schema.json"
    schema = json.loads(path.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_json_schema_validates_regular_zero_reference_and_unavailable_metrics():
    validator = comparison_validator()
    validator.validate(compare())
    validator.validate(compare(reference={"u": [0, 0, 0]}))
    for predictions, references in (
        ([1], [math.ulp(0.0)]),
        ([2, math.ulp(0.0)], [2, 0]),
        ([math.ulp(0.0), 0, 0, 0, 0], [0, 0, 0, 0, 0]),
    ):
        report = compare(
            coordinates={"x": list(range(len(predictions)))},
            candidate={"u": predictions},
            reference={"u": references},
        )
        validator.validate(report)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "PROVED"},
        {"evidence_level": "EXACT"},
        {"sample_count": 0},
        {"input_sha256": "not-a-digest"},
        {"reference_comparison_version": 2},
    ],
)
def test_json_schema_rejects_unsupported_report_claims_and_shape(changes):
    report = compare()
    report.update(changes)
    with pytest.raises(ValidationError):
        comparison_validator().validate(report)


@pytest.mark.parametrize(
    "changes",
    [
        {"rmse": -1},
        {"relative_l2": -1},
        {"max_absolute_error": -1},
        {"rmse": None, "rmse_unavailable_reason": None},
        {"rmse": 0, "rmse_unavailable_reason": "binary64_underflow"},
        {"relative_l2": None, "relative_l2_unavailable_reason": None},
        {"relative_l2": 0, "relative_l2_unavailable_reason": "zero_reference_norm"},
        {"status": "REFUTED"},
    ],
)
def test_json_schema_rejects_inconsistent_metric_availability(changes):
    report = compare()
    report["fields"][0].update(changes)
    with pytest.raises(ValidationError):
        comparison_validator().validate(report)


def test_fisher_kpp_example_is_a_reproducible_injected_field_perturbation():
    from examples.reference_field_comparison import run

    report = run()
    assert report == run()
    assert report["sample_count"] == 25
    assert report["evidence_level"] == "EMPIRICAL"
    assert "perturbation" in report["candidate_id"]
    assert report["reference"]["assessment"] == "caller_supplied_not_verified"
    field = report["fields"][0]
    times = [0.1, 0.55, 1, 1.45, 1.9]
    expected_rmse = 0.02 * math.sqrt(sum(time**2 for time in times) / len(times))
    assert field["rmse"] == pytest.approx(expected_rmse, rel=1e-14)
    assert field["relative_l2"] == pytest.approx(0.04141737408527671, rel=1e-14)
    assert field["max_absolute_error"] == pytest.approx(0.038, rel=1e-14)
    assert field["max_error_sample"]["coordinates"]["t"] == 1.9
    assert field["max_error_sample"]["index"] % 5 == 4
    comparison_validator().validate(report)
    json.dumps(report, allow_nan=False)
