"""Offline consumer regressions; mocked Hub checks are not live integration evidence."""

import copy
import hashlib
import importlib
import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import hub_pilot_smoke as smoke
from pdecert import (
    CorpusError,
    build_release_bundle,
    corpus_sha256,
    load_corpus,
    output_sha256,
    report_from_dict,
    validate_corpus,
)


@pytest.fixture(scope="module")
def pilot():
    return load_corpus(smoke.ROOT / "corpus/pilot.json")


@pytest.fixture
def synthetic():
    """Independent pending fixtures exercise proof, refutation, and abstention."""
    records = []
    for name, expression in (("exact", "0"), ("defect", "x"), ("tiny", "x/100000000000000")):
        raw = f"synthetic test expression: {expression}"
        records.append(
            {
                "id": f"synthetic-{name}",
                "raw_output": raw,
                "output_sha256": output_sha256(raw),
                "origin": {
                    "kind": "synthetic",
                    "identifier": f"test-{name}",
                    "producer": "offline test fixture",
                    "input": "Solve D(u, x) = 0.",
                    "license": None,
                    "revision": None,
                    "version": None,
                    "source_url": "https://example.invalid/synthetic-fixture",
                    "generated_at": "2026-01-01T00:00:00Z",
                },
                "annotation": {
                    "annotators": [],
                    "failure_modes": [],
                    "rationale": None,
                    "status": "pending",
                    "verdict": None,
                },
                "case": {
                    "schema_version": 3,
                    "name": f"constant-function test {name}",
                    "variables": ["x"],
                    "domains": {"x": [0.0, 1.0]},
                    "parameters": {},
                    "fields": {"u": expression},
                    "pde_residuals": [{"name": "constant derivative", "expression": "D(u, x)"}],
                    "conditions": [],
                },
            }
        )
    corpus = {
        "corpus_version": 1,
        "name": "independent consumer fixtures",
        "description": "Synthetic pending examples; not human-labeled pilot records.",
        "records": records,
    }
    validate_corpus(corpus)
    return corpus


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _replace(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def test_strict_roundtrip_preserves_all_nullable_values_and_input_objects(synthetic):
    rows = copy.deepcopy(synthetic["records"])
    source_before, rows_before = _json(synthetic), _json(rows)
    restored = smoke.check_loaded_rows(synthetic, rows)

    assert _json(restored) == source_before
    assert _json(synthetic) == source_before
    assert _json(rows) == rows_before
    assert corpus_sha256(restored) == corpus_sha256(synthetic)
    for row in restored["records"]:
        assert all(row["origin"][key] is None for key in ("license", "revision", "version"))
        assert row["annotation"]["rationale"] is None
        assert row["annotation"]["verdict"] is None
    restored["records"][0]["case"]["domains"]["x"][0] = -1.0
    assert _json(synthetic) == source_before
    assert _json(rows) == rows_before


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("case", "fields", "u"), "0 + 0"),
        (("case", "domains", "x"), [-1.0, 1.0]),
        (("case", "domains", "y"), None),
        (("case", "parameters", "x"), None),
        (("case", "parameters", "x"), []),
        (("case", "pde_residuals", 0, "expression"), "u"),
        (("case", "variables"), ["y"]),
        (("case", "schema_version"), 3.0),
        (("raw_output",), "changed output"),
        (("output_sha256",), "0" * 64),
        (("origin", "revision"), "changed revision"),
        (("origin", "input"), "changed prompt"),
        (("annotation", "status"), "labeled"),
        (("annotation", "verdict"), "valid"),
        (("annotation", "rationale"), "changed rationale"),
        (("annotation", "annotators"), ["synthetic-reviewer"]),
        (("annotation", "failure_modes"), ["pde_residual"]),
        (("unexpected",), None),
    ],
)
def test_nested_transport_changes_fail_without_mutation(synthetic, path, replacement):
    rows = copy.deepcopy(synthetic["records"])
    _replace(rows[0], path, replacement)
    source_before, rows_before = _json(synthetic), _json(rows)
    with pytest.raises(smoke.HubRoundtripError, match="loaded content differs"):
        smoke.check_loaded_rows(synthetic, rows)
    assert _json(synthetic) == source_before
    assert _json(rows) == rows_before


@pytest.mark.parametrize("field", ["license", "revision", "version"])
def test_removing_required_nullable_origin_field_is_refused(synthetic, field):
    rows = copy.deepcopy(synthetic["records"])
    del rows[0]["origin"][field]
    with pytest.raises(smoke.HubRoundtripError, match="loaded content differs"):
        smoke.check_loaded_rows(synthetic, rows)


def test_changed_raw_output_is_refused_even_with_matching_new_digest(synthetic):
    rows = copy.deepcopy(synthetic["records"])
    rows[0]["raw_output"] += " changed"
    rows[0]["output_sha256"] = output_sha256(rows[0]["raw_output"])
    with pytest.raises(smoke.HubRoundtripError, match="loaded content differs"):
        smoke.check_loaded_rows(synthetic, rows)


@pytest.mark.parametrize("change", ["drop", "append", "duplicate", "reorder", "id"])
def test_row_count_identity_and_order_are_required(synthetic, change):
    rows = copy.deepcopy(synthetic["records"])
    if change == "drop":
        rows.pop()
    elif change == "append":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif change == "reorder":
        rows.reverse()
    else:
        rows[0]["id"] = "different-id"
    with pytest.raises(smoke.HubRoundtripError, match="row count|loaded content differs"):
        smoke.check_loaded_rows(synthetic, rows)


@pytest.mark.parametrize(("original", "replacement"), [(0.0, 0), (0, False), (1, True)])
def test_python_equal_number_changes_are_not_json_equal(synthetic, original, replacement):
    bound = 1 if original == 1 else 0
    synthetic["records"][0]["case"]["domains"]["x"][bound] = original
    rows = copy.deepcopy(synthetic["records"])
    rows[0]["case"]["domains"]["x"][bound] = replacement
    assert rows == synthetic["records"]
    with pytest.raises(smoke.HubRoundtripError, match="loaded content differs"):
        smoke.check_loaded_rows(synthetic, rows)


@pytest.mark.parametrize(
    ("path", "unsupported"),
    [
        (("corpus_version",), 999),
        (("corpus_version",), True),
        (("records", 0, "case", "schema_version"), 1),
        (("records", 0, "case", "schema_version"), 2),
        (("records", 0, "case", "schema_version"), 4),
    ],
)
def test_reference_corpus_is_validated_before_identical_rows_are_accepted(
    synthetic, path, unsupported
):
    _replace(synthetic, path, unsupported)
    with pytest.raises(CorpusError, match="expected"):
        smoke.check_loaded_rows(synthetic, copy.deepcopy(synthetic["records"]))


def test_real_verifier_keeps_synthetic_empirical_pass_inconclusive(synthetic):
    before = _json(synthetic)
    results = smoke.evaluate_loaded_rows(synthetic)
    assert [result["id"] for result in results] == [row["id"] for row in synthetic["records"]]
    exact, defect, tiny = [result["report"] for result in results]
    assert [report["status"] for report in (exact, defect, tiny)] == [
        "PROVED",
        "REFUTED",
        "INCONCLUSIVE",
    ]
    assert exact["decision_evidence"] == "EXACT"
    assert defect["witness"] is not None
    assert tiny["decision_evidence"] is None
    passes = [event for event in tiny["evidence_events"] if event["kind"] == "EMPIRICAL_PASS"]
    assert passes
    assert all(event["outcome"] == "OBSERVED_PASS" for event in passes)
    for result in results:
        assert report_from_dict(result["report"]).to_dict() == result["report"]
    assert _json(synthetic) == before


def test_real_installed_datasets_local_offline_roundtrip(pilot, tmp_path, monkeypatch):
    """Real optional Datasets integration, with no verifier or loader mocks."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as error:
        if error.name == "datasets":
            pytest.skip("optional datasets dependency is not installed")
        raise
    hub_constants = importlib.import_module("huggingface_hub.constants")
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)

    attempted_connections = []

    def reject_network(*args, **kwargs):
        attempted_connections.append((args, kwargs))
        raise AssertionError("local Datasets roundtrip attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    benchmark = json.loads((smoke.ROOT / "results/pilot-benchmark.json").read_text())
    bundle = tmp_path / "bundle"
    manifest = build_release_bundle(pilot, benchmark, bundle)
    source = bundle / smoke.DATA_FILE
    before = _json(pilot)
    dataset = datasets.load_dataset(
        "json",
        name=smoke.CONFIG,
        data_files={smoke.SPLIT: str(source)},
        split=smoke.SPLIT,
        cache_dir=str(tmp_path / "datasets"),
        token=False,
    )
    assert dataset.info.config_name == smoke.CONFIG
    assert str(dataset.split) == smoke.SPLIT
    assert dataset.num_rows == 20
    assert set(dataset.column_names) == set(pilot["records"][0])
    rebuilt = smoke.check_loaded_rows(pilot, list(dataset))
    assert _json(rebuilt) == before
    assert _json(pilot) == before
    assert corpus_sha256(rebuilt) == smoke.PILOT_SHA256
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"][smoke.DATA_FILE]

    # Exercise the public runner too, including every loaded case's real verifier.
    # Start with both active switches off to test its own offline guard.
    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    result = smoke.run_smoke()
    assert datasets.config.HF_HUB_OFFLINE is False
    assert hub_constants.HF_HUB_OFFLINE is False
    assert attempted_connections == []  # Includes attempts swallowed by a library.
    assert result["mode"] == "local_jsonl"
    assert result["dataset"]["repo_id"] is None
    assert result["dataset"]["revision"] is None
    assert result["dataset"]["jsonl_sha256"] == manifest["files"][smoke.DATA_FILE]
    assert result["dataset"]["corpus_sha256"] == smoke.PILOT_SHA256
    assert sum(result["status_counts"].values()) == 20
    assert [item["id"] for item in result["records"]] == [row["id"] for row in pilot["records"]]
    for item in result["records"]:
        assert report_from_dict(item["report"]).to_dict() == item["report"]

    with pytest.raises(RuntimeError, match="synthetic loader error"):
        with smoke._offline_loading():
            assert datasets.config.HF_HUB_OFFLINE is True
            assert hub_constants.HF_HUB_OFFLINE is True
            raise RuntimeError("synthetic loader error")
    assert datasets.config.HF_HUB_OFFLINE is False
    assert hub_constants.HF_HUB_OFFLINE is False


@pytest.fixture
def mocked_hub(pilot, tmp_path, monkeypatch):
    """Fake modules test Hub call arguments and guards without network or optional packages."""
    source = tmp_path / "mock-hub-pilot.jsonl"
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pilot["records"]))

    class MockDataset(list):
        pass

    dataset = MockDataset(copy.deepcopy(pilot["records"]))
    dataset.info = SimpleNamespace(config_name=smoke.CONFIG)
    dataset.split = smoke.SPLIT
    dataset.num_rows = len(dataset)
    dataset.column_names = list(pilot["records"][0])
    loader = Mock(return_value=dataset)
    download = Mock(return_value=str(source))
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = loader
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setattr(smoke, "version", lambda package: "mock-test-version")
    reports = [{"id": row["id"], "report": {"status": "INCONCLUSIVE"}} for row in dataset]
    evaluate = Mock(return_value=reports)
    monkeypatch.setattr(smoke, "evaluate_loaded_rows", evaluate)
    return SimpleNamespace(
        source=source, dataset=dataset, loader=loader, download=download, evaluate=evaluate
    )


def test_mocked_hub_uses_exact_public_revision_config_split_and_anonymous_access(mocked_hub, pilot):
    result = smoke.run_smoke(from_hub=True)
    assert mocked_hub.download.call_count == mocked_hub.loader.call_count == 1
    args, kwargs = mocked_hub.download.call_args
    assert args == (smoke.PILOT_REPO, smoke.DATA_FILE)
    assert set(kwargs) == {"repo_type", "revision", "cache_dir", "token"}
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "db690f9b161762ea288dd5dfb4b6b2f999c48e03"
    assert kwargs["token"] is False
    assert Path(kwargs["cache_dir"]).name == "hub"
    args, kwargs = mocked_hub.loader.call_args
    assert args == (smoke.PILOT_REPO,)
    assert set(kwargs) == {"name", "split", "revision", "cache_dir", "token"}
    assert kwargs["name"] == "default"
    assert kwargs["split"] == "test"
    assert kwargs["revision"] == smoke.PILOT_REVISION
    assert kwargs["token"] is False
    assert Path(kwargs["cache_dir"]).name == "datasets"
    mocked_hub.evaluate.assert_called_once_with(pilot)
    assert result["mode"] == "hub"
    assert result["dataset"]["corpus_sha256"] == smoke.PILOT_SHA256
    assert result["dataset"]["record_count"] == 20
    assert result["status_counts"] == {"INCONCLUSIVE": 20}
    assert result["environment"]["datasets"] == "mock-test-version"


def test_mocked_hub_download_byte_corruption_fails_before_loading_or_verification(mocked_hub):
    # A harmless extra newline preserves rows but violates the frozen file's byte identity.
    mocked_hub.source.write_bytes(mocked_hub.source.read_bytes() + b"\n")
    with pytest.raises(smoke.HubRoundtripError, match="JSONL bytes differ"):
        smoke.run_smoke(from_hub=True)
    mocked_hub.loader.assert_not_called()
    mocked_hub.evaluate.assert_not_called()


@pytest.mark.parametrize("guard", ["config", "split", "count", "columns", "rows"])
def test_mocked_hub_loaded_metadata_and_row_corruption_are_refused(mocked_hub, guard):
    dataset = mocked_hub.dataset
    if guard == "config":
        dataset.info.config_name = "unexpected"
    elif guard == "split":
        dataset.split = "train"
    elif guard == "count":
        dataset.num_rows -= 1
    elif guard == "columns":
        dataset.column_names.append("unexpected")
    else:
        dataset[0]["case"]["domains"]["y"] = None
    with pytest.raises(smoke.HubRoundtripError, match="config or split|row count|columns|content"):
        smoke.run_smoke(from_hub=True)
    mocked_hub.evaluate.assert_not_called()


def test_mocked_transport_refuses_changed_source_before_any_hub_read(
    mocked_hub, pilot, monkeypatch
):
    changed = copy.deepcopy(pilot)
    changed["description"] += " changed"
    monkeypatch.setattr(smoke, "load_corpus", lambda path: changed)
    with pytest.raises(smoke.HubRoundtripError, match="frozen pilot digest"):
        smoke.run_smoke(from_hub=True)
    mocked_hub.download.assert_not_called()
    mocked_hub.loader.assert_not_called()
    mocked_hub.evaluate.assert_not_called()


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(smoke.HubRoundtripError, match="requirements/hub-integration.txt"):
        smoke.run_smoke()


def test_cli_writes_new_json_output_and_refuses_existing_file(tmp_path, monkeypatch, capsys):
    run = Mock(return_value={"scope": "mocked CLI output fixture"})
    monkeypatch.setattr(smoke, "run_smoke", run)
    destination = tmp_path / "mock-report.json"
    assert smoke.main(["--from-hub", "--output", str(destination)]) == 0
    run.assert_called_once_with(from_hub=True)
    assert json.loads(destination.read_text()) == run.return_value
    original = destination.read_bytes()
    run.reset_mock()
    assert smoke.main(["--output", str(destination)]) == 1
    run.assert_not_called()
    assert destination.read_bytes() == original
    assert "output already exists" in capsys.readouterr().err


def test_cli_failure_does_not_create_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run_smoke", Mock(side_effect=smoke.HubRoundtripError("bad rows")))
    destination = tmp_path / "failed-report.json"
    assert smoke.main(["--output", str(destination)]) == 1
    assert not destination.exists()
    captured = capsys.readouterr()
    assert "bad rows" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out
