import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from pdecert import ReferenceComparisonError, compare_reference_fields, compare_reference_file
from pdecert import reference_io
from pdecert.cli import INPUT_ERROR, main

ROOT = Path(__file__).resolve().parents[1]


def sample_input():
    return {
        "reference_input_version": 1,
        "coordinates": {"x": [0, 0.5, 1]},
        "candidate": {"u": [1, 2, 4]},
        "reference": {"u": [1, 2, 3]},
        "problem_id": "supplied-problem",
        "candidate_id": "supplied-candidate",
        "reference_id": "supplied-reference",
        "reference_description": "Caller-supplied comparison values",
        "reference_uncertainty": "No validated uncertainty estimate is available",
        "sampling_description": "Three equally weighted supplied rows",
    }


def write_input(tmp_path, **changes):
    payload = sample_input()
    payload.update(changes)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def input_validator():
    schema = json.loads((ROOT / "schema/reference-input-v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_file_comparison_preserves_the_direct_report(tmp_path):
    payload = sample_input()
    payload.pop("reference_input_version")
    report = compare_reference_file(write_input(tmp_path))
    assert report == compare_reference_fields(**payload)
    assert report["fields"][0]["rmse"] == pytest.approx(math.sqrt(1 / 3))
    assert report["fields"][0]["max_error_sample"]["index"] == 2
    assert reference_io.REFERENCE_INPUT_VERSION == 1
    assert reference_io.REFERENCE_INPUT_MAX_BYTES == 64 * 1024 * 1024


def test_shipped_fixture_is_structurally_valid_and_reproducible():
    path = ROOT / "examples/reference-input.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    input_validator().validate(payload)
    assert payload["coordinates"] == {"x": [0, 0.5, 1]}
    assert payload["candidate"] == {"u": [1, 2, 4]}
    assert payload["reference"] == {"u": [1, 2, 3]}
    payload.pop("reference_input_version")
    assert compare_reference_file(path) == compare_reference_fields(**payload)


@pytest.mark.parametrize("version", [True, 1.0, 0, 2, "1", None])
def test_version_requires_the_supported_json_integer(tmp_path, version):
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(write_input(tmp_path, reference_input_version=version))


@pytest.mark.parametrize("key", list(sample_input()))
def test_every_envelope_field_is_required(tmp_path, key):
    payload = sample_input()
    del payload[key]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": "value"},
        {"candidate": {}},
        {"coordinates": []},
        {"candidate": {"u": [[1], [2], [4]]}},
        {"candidate": {"u": [True, 2, 4]}},
        {"reference_description": "  "},
        {"problem_id": "\ud800"},
        {"coordinates": {"\udfff": [0, 0.5, 1]}},
    ],
)
def test_invalid_envelope_and_column_values_are_rejected(tmp_path, changes):
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(write_input(tmp_path, **changes))


@pytest.mark.parametrize(
    "raw", [b"[]", b"null", b"{", b"{} {}", b"\xff", b"[" * 1500 + b"0" + b"]" * 1500]
)
def test_nonobjects_malformed_trailing_deep_json_and_invalid_utf8(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(path)


def test_nonregular_input_is_rejected(tmp_path):
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(tmp_path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "9007199254740993"])
def test_json_numbers_must_be_finite_and_exactly_representable_integers(tmp_path, token):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(sample_input()).replace("[1, 2, 4]", f"[{token}, 2, 4]"))
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(path)


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_keys_are_rejected_at_every_object_depth(tmp_path, nested):
    raw = json.dumps(sample_input())
    raw = (
        raw.replace('"u": [1, 2, 4]', '"u": [1, 2, 4], "u": [1, 2, 4]')
        if nested
        else (
            raw.replace(
                '"reference_input_version": 1',
                '"reference_input_version": 1, "reference_input_version": 1',
            )
        )
    )
    path = tmp_path / "input.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(path)


def test_byte_budget_accepts_the_limit_and_rejects_one_more_byte(tmp_path, monkeypatch):
    path = write_input(tmp_path)
    size = path.stat().st_size
    monkeypatch.setattr(reference_io, "REFERENCE_INPUT_MAX_BYTES", size)
    assert compare_reference_file(path)["sample_count"] == 3
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate": {"v": [1, 2, 4]}},
        {"reference": {"u": [1, 2]}},
        {"reference_input_version": 1.0},
        {"candidate": {"u": [9007199254740993, 2, 4]}},
    ],
)
def test_schema_is_structural_and_runtime_checks_cross_field_and_numeric_rules(tmp_path, changes):
    payload = sample_input()
    payload.update(changes)
    input_validator().validate(payload)
    with pytest.raises(ReferenceComparisonError):
        compare_reference_file(write_input(tmp_path, **changes))


@pytest.mark.parametrize(
    "changes",
    [{"extra": 1}, {"candidate": {}}, {"candidate": {"u": [True]}}, {"reference_input_version": 2}],
)
def test_schema_rejects_invalid_envelope_shapes(changes):
    payload = sample_input()
    payload.update(changes)
    with pytest.raises(ValidationError):
        input_validator().validate(payload)


@pytest.mark.parametrize("reference", [{"u": [1, 2, 3]}, {"u": [0, 0, 0]}])
def test_cli_success_is_empirical_even_with_large_error_or_undefined_relative(
    tmp_path, capsys, reference
):
    path = write_input(tmp_path, candidate={"u": [1e100, 2, 4]}, reference=reference)
    assert main(["compare-reference", str(path)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report == compare_reference_file(path)
    assert report["evidence_level"] == "EMPIRICAL"
    assert "status" not in report
    if not any(reference["u"]):
        assert report["fields"][0]["relative_l2"] is None
        assert report["fields"][0]["relative_l2_unavailable_reason"] == "zero_reference_norm"


@pytest.mark.parametrize("option", ["-o", "--output"])
def test_cli_writes_a_report_file(tmp_path, capsys, option):
    source = write_input(tmp_path)
    output = tmp_path / "report.json"
    assert main(["compare-reference", str(source), option, str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text()) == compare_reference_file(source)


@pytest.mark.parametrize("failure", ["missing", "malformed", "output_directory"])
def test_cli_errors_are_concise_and_invalid_input_preserves_output(tmp_path, capsys, failure):
    source = write_input(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("existing report")
    if failure == "missing":
        source = tmp_path / "missing.json"
    elif failure == "malformed":
        source.write_text("{")
    destination = tmp_path if failure == "output_directory" else output
    assert main(["compare-reference", str(source), "-o", str(destination)]) == INPUT_ERROR == 64
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err and "Traceback" not in captured.err
    assert output.read_text() == "existing report"


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_cli_cannot_overwrite_its_source_through_an_alias(tmp_path, capsys, alias):
    source = write_input(tmp_path)
    before = source.read_bytes()
    output = source if alias == "same" else tmp_path / "alias.json"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        os.link(source, output)
    assert main(["compare-reference", str(source), "-o", str(output)]) == 64
    assert source.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err


def test_public_file_api_works_from_empty_cwd_without_model_imports(tmp_path):
    source = write_input(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    script = (
        "import sys; from pdecert import compare_reference_file; "
        "assert compare_reference_file(sys.argv[1])['sample_count'] == 3; "
        "assert not {'torch', 'mlx', 'transformers', 'smolagents'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        cwd=empty,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("loop_at", ["input", "output"])
def test_cli_reports_symlink_loops_as_file_errors(tmp_path, capsys, loop_at):
    source = write_input(tmp_path)
    before = source.read_bytes()
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop.name)
    arguments = (
        ["compare-reference", str(loop)]
        if loop_at == "input"
        else ["compare-reference", str(source), "--output", str(loop)]
    )
    assert main(arguments) == INPUT_ERROR
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err and "Traceback" not in captured.err
    assert source.read_bytes() == before
