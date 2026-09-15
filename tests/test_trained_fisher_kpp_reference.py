"""The trained-field example retains empirical, separately scoped evidence."""

import builtins
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from jsonschema import Draft202012Validator

from pdecert import (
    CallableCandidate,
    FrozenCallableError,
    Status,
    compare_reference_fields,
    compile_autodiff_problem,
    frozen_callable_to_dict,
    report_from_dict,
)

from experiments import trained_fisher_kpp_reference as example


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("benchmarks/matched/fisher-kpp-classical-01/pinn.json")
TEMPLATE = Path("benchmarks/matched/fisher-kpp-classical-01/template.json")
INTEGRITY = Path("benchmarks/matched/fisher-kpp-classical-01/integrity.json")
TEMPLATE_SHA256 = "8d6248c31072d4bc7e109f697ce2637c00420683a68edab7ae99624bc1f6d4a6"
FRACTIONS = (0.113, 0.271, 0.419, 0.613, 0.787, 0.937)


@pytest.fixture
def copied_root(tmp_path):
    """Copy only the bundled fixture, bound inputs, and this example's sources."""

    integrity = json.loads((ROOT / INTEGRITY).read_text())
    files = {
        FIXTURE,
        INTEGRITY,
        Path("experiments/trained_fisher_kpp_reference.py"),
        Path("src/pdecert/reference_fields.py"),
        Path("schema/reference-comparison-v1.schema.json"),
        *(Path(path) for path in integrity["source_files_sha256"]),
    }
    for relative in files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch", reason="trained reference example requires PyTorch")


@pytest.fixture(scope="module")
def result(torch):
    return example.run(repository_root=ROOT)


def test_inputs_bind_the_existing_frozen_artifact_without_materializing(monkeypatch):
    materialize = Mock(side_effect=AssertionError("input validation must not execute the model"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    template, frozen, integrity = example.load_inputs(ROOT)
    assert template.variables == ("x", "t")
    assert template.field_names == ("u",)
    assert template.solution_semantics == "classical_strong"
    assert frozen_callable_to_dict(frozen)["problem_id"] == "fisher-kpp-classical-01"
    assert integrity == json.loads((ROOT / INTEGRITY).read_text())
    assert hashlib.sha256((ROOT / TEMPLATE).read_bytes()).hexdigest() == TEMPLATE_SHA256
    materialize.assert_not_called()


@pytest.mark.parametrize(
    "relative",
    [FIXTURE, TEMPLATE, Path("src/pdecert/autodiff.py")],
)
def test_altered_bound_bytes_are_rejected_before_model_execution(
    copied_root, monkeypatch, relative
):
    path = copied_root / relative
    path.write_bytes(path.read_bytes() + b"\n")
    materialize = Mock(side_effect=AssertionError("corrupted input reached model execution"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError, match="digest|template|reference"):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


@pytest.mark.parametrize("field", ["configuration_sha256", "weights_sha256", "artifact_sha256"])
def test_manifest_digest_mismatch_prevents_model_execution(copied_root, monkeypatch, field):
    path = copied_root / INTEGRITY
    integrity = json.loads(path.read_text())
    integrity[field] = "0" * 64
    path.write_text(json.dumps(integrity))
    materialize = Mock(side_effect=AssertionError("wrong digest reached model execution"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError, match="digest|integrity"):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_self_consistent_wrong_template_cannot_rebind_the_analytical_reference(
    copied_root, monkeypatch
):
    template_path = copied_root / TEMPLATE
    template = json.loads(template_path.read_text())
    template["pde_residuals"][0]["expression"] = "D(u, t) - D(u, x, 2) - 2*u*(1-u)"
    template_path.write_text(json.dumps(template))
    integrity_path = copied_root / INTEGRITY
    integrity = json.loads(integrity_path.read_text())
    integrity["source_files_sha256"][TEMPLATE.as_posix()] = hashlib.sha256(
        template_path.read_bytes()
    ).hexdigest()
    integrity_path.write_text(json.dumps(integrity))
    materialize = Mock(side_effect=AssertionError("unbound reference reached model execution"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError, match="template|reference|integrity"):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_unsupported_frozen_architecture_is_rejected_before_execution(copied_root, monkeypatch):
    fixture_path = copied_root / FIXTURE
    fixture = json.loads(fixture_path.read_text())
    fixture["architecture"]["activation"] = "relu"
    fixture_path.write_text(json.dumps(fixture))
    materialize = Mock(side_effect=AssertionError("unsupported architecture was executed"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_missing_bound_source_is_rejected_before_execution(copied_root, monkeypatch):
    (copied_root / "src/pdecert/autodiff.py").unlink()
    materialize = Mock(side_effect=AssertionError("missing source validation was bypassed"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises((FrozenCallableError, FileNotFoundError)):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_omitted_evaluator_source_binding_is_rejected_before_execution(copied_root, monkeypatch):
    integrity_path = copied_root / INTEGRITY
    integrity = json.loads(integrity_path.read_text())
    del integrity["source_files_sha256"]["src/pdecert/autodiff.py"]
    integrity_path.write_text(json.dumps(integrity))
    materialize = Mock(side_effect=AssertionError("unbound evaluator source reached execution"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError, match="autodiff|integrity"):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_declared_source_copy_cannot_substitute_for_the_imported_evaluator(
    copied_root, monkeypatch
):
    relative = "src/pdecert/autodiff.py"
    source_path = copied_root / relative
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    integrity_path = copied_root / INTEGRITY
    integrity = json.loads(integrity_path.read_text())
    integrity["source_files_sha256"][relative] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    integrity_path.write_text(json.dumps(integrity))
    materialize = Mock(side_effect=AssertionError("unbound imported evaluator reached execution"))
    monkeypatch.setattr(example, "materialize_frozen_callable", materialize)
    with pytest.raises(FrozenCallableError, match="imported source"):
        example.run(repository_root=copied_root)
    materialize.assert_not_called()


def test_missing_torch_explains_the_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is deliberately unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(RuntimeError, match="autodiff"):
        example.run(repository_root=ROOT)


def test_sampling_replays_the_existing_checker_order_and_fixed_surfaces():
    template, _, _ = example.load_inputs(ROOT)
    problem = compile_autodiff_problem(template)
    xs = [-6 + 12 * fraction for fraction in FRACTIONS]
    ts = [2 * fraction for fraction in FRACTIONS]
    expected = [
        {"x": [x for x in xs for _ in ts], "t": [t for _ in xs for t in ts]},
        {"x": xs, "t": [0.0] * 6},
        {"x": [-6.0] * 6, "t": ts},
        {"x": [6.0] * 6, "t": ts},
    ]
    for constraint, coordinates in zip(problem.constraints, expected, strict=True):
        assert example.sample_coordinates(problem, constraint) == coordinates


def test_every_obligation_has_separate_empirical_evidence_even_after_pde_failure(result):
    assert result["suite"] == "trained-fisher-kpp-reference-v1"
    assert "status" not in result
    assert "label" not in result
    assert result["evaluation"]["dtype"] == "float64"
    assert result["evaluation"]["device"] == "cpu"
    assert result["evaluation"]["samples_per_axis"] == 6
    obligations = result["obligations"]
    assert [row["source_obligation_id"] for row in obligations] == [
        f"constraint:{index}" for index in range(4)
    ]
    assert [row["region"] for row in obligations] == [
        "interior",
        "initial",
        "left_boundary",
        "right_boundary",
    ]
    assert [row["reference_comparison"]["sample_count"] for row in obligations] == [36, 6, 6, 6]
    assert obligations[0]["diagnostic_report"]["status"] == "REFUTED"
    assert obligations[0]["diagnostic_report"]["decision_evidence"] == "EMPIRICAL"
    assert [row["diagnostic_report"]["status"] for row in obligations] == [
        "REFUTED",
        "REFUTED",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
    ]
    assert "PROVED" not in json.dumps(obligations)
    for row in obligations:
        report = report_from_dict(row["diagnostic_report"])
        assert report.status in {Status.REFUTED, Status.INCONCLUSIVE}
        assert report.exact_checks == {}
        assert report.evidence_events
        assert {event.obligation_id for event in report.evidence_events} == {"constraint:0"}
        assert row["reference_comparison"]["evidence_level"] == "EMPIRICAL"
        assert (
            row["reference_comparison"]["reference"]["assessment"] == "caller_supplied_not_verified"
        )
        assert row["reference_comparison"]["reference"]["uncertainty"]
        if report.witness is not None:
            coordinates = row["samples"]["coordinates"]
            sampled_points = [
                {"x": x, "t": t} for x, t in zip(coordinates["x"], coordinates["t"], strict=True)
            ]
            assert report.witness.point in sampled_points
            assert report.witness.residual > result["evaluation"]["tolerance"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_report_preserves_frozen_training_identity_and_reproduction_metadata(result):
    _, frozen, integrity = example.load_inputs(ROOT)
    payload = frozen_callable_to_dict(frozen)
    assert result["integrity"] == integrity
    for field in ("configuration_sha256", "weights_sha256"):
        assert result["fixture"][field] == integrity[field]
    assert result["fixture"]["artifact_sha256"] == integrity["artifact_sha256"]
    assert result["fixture"]["artifact_id"] == payload["artifact_id"]
    assert result["fixture"]["training"] == payload["training"]
    for relative, digest in result["source_digests"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert "experiments/trained_fisher_kpp_reference.py" in result["source_digests"]
    assert "src/pdecert/reference_fields.py" in result["source_digests"]
    assert result["runtime"]["torch_version"]
    assert result["runtime"]["python_version"]
    assert result["runtime"]["pdecert_version"]
    assert result["evaluation"]["training_performed"] is False


def test_all_reference_reports_validate_and_replay_from_their_published_samples(result):
    schema = json.loads((ROOT / "schema/reference-comparison-v1.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for row in result["obligations"]:
        comparison = row["reference_comparison"]
        validator.validate(comparison)
        replayed = compare_reference_fields(
            **row["samples"],
            problem_id=comparison["problem_id"],
            candidate_id=comparison["candidate_id"],
            reference_id=comparison["reference"]["id"],
            reference_description=comparison["reference"]["description"],
            reference_uncertainty=comparison["reference"]["uncertainty"],
            sampling_description=comparison["sampling_description"],
        )
        assert replayed == comparison
        candidate = row["samples"]["candidate"]["u"]
        reference = row["samples"]["reference"]["u"]
        errors = [c - r for c, r in zip(candidate, reference, strict=True)]
        metrics = comparison["fields"][0]
        assert metrics["rmse"] == pytest.approx(
            math.sqrt(math.fsum(e * e for e in errors) / len(errors))
        )
        assert metrics["relative_l2"] == pytest.approx(
            math.sqrt(math.fsum(e * e for e in errors) / math.fsum(r * r for r in reference))
        )
        assert metrics["max_absolute_error"] == max(abs(error) for error in errors)
        coordinates = row["samples"]["coordinates"]
        for x, t, value in zip(coordinates["x"], coordinates["t"], reference, strict=True):
            assert value == pytest.approx((1 + math.exp(x / math.sqrt(6) - 5 * t / 6)) ** -2)


def test_repeated_evaluation_keeps_sample_and_metric_identities(result, torch):
    repeated = example.run(repository_root=ROOT)
    assert repeated == result


def test_injected_analytical_control_remains_inconclusive_not_proved(torch, monkeypatch):
    """A test-only exact-form replacement is not a newly trained artifact."""

    def front(points):
        x = points[:, 0:1]
        t = points[:, 1:2]
        return (1 + torch.exp(x / math.sqrt(6) - 5 * t / 6)) ** -2

    control = CallableCandidate.from_mapping({"u": front}, dtype="float64", device="cpu")
    monkeypatch.setattr(example, "materialize_frozen_callable", lambda _: control)
    control_result = example.run(repository_root=ROOT)
    for row in control_result["obligations"]:
        assert row["diagnostic_report"]["status"] == "INCONCLUSIVE"
        assert row["diagnostic_report"]["decision_evidence"] is None
        assert row["reference_comparison"]["fields"][0]["max_absolute_error"] < 1e-14
    assert "PROVED" not in json.dumps(control_result["obligations"])


def test_cli_works_from_outside_the_checkout(tmp_path, result, torch):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Earlier training tests may set the parent process to one thread. Replay
    # that declared setting rather than compare it to the child CPU default.
    assert result["runtime"]["torch_num_threads"] > 0
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = str(result["runtime"]["torch_num_threads"])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.trained_fisher_kpp_reference",
            "--repository-root",
            str(ROOT),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(completed.stdout) == result
