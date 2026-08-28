import json
import shutil

import pytest

from pdecert import (
    RunManifestError,
    bind_symbolic_candidate,
    build_run_manifest,
    dump_run_manifest,
    load_run_manifest,
    load_template,
    manifest_from_dict,
    manifest_to_dict,
    run_manifest_sha256,
    validate_run_bundle,
    verify,
)


def _write_bundle(directory):
    template_path = directory / "template.json"
    candidate_path = directory / "candidate.json"
    report_path = directory / "report.json"
    shutil.copyfile("examples/heat-template.json", template_path)
    candidate_path.write_text(
        json.dumps(
            {"fields": {"u": "exp(-pi**2*t)*sin(pi*x)"}},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    template = load_template(template_path)
    case = bind_symbolic_candidate(template, {"u": "exp(-pi**2*t)*sin(pi*x)"})
    report = verify(case.problem, case.candidate_fields)
    report_path.write_text(
        json.dumps(
            {"problem": template.name, "report": report.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = build_run_manifest(
        bundle_root=directory,
        run_id="heat-exact-symbolic-01",
        problem_id="heat-classical-01",
        template_path=template_path,
        candidate_path=candidate_path,
        report_path=report_path,
        artifact_id="exact-heat-expression-01",
        artifact_kind="symbolic",
        field_names=("u",),
        provenance={"producer": "PDECert example", "revision": "version 1"},
        evaluator_name="pdecert.symbolic",
        evaluator_version="0.1.1rc1",
        evaluator_configuration={
            "max_expression_ops": 10_000,
            "samples_per_axis": 5,
            "symbolic_timeout_seconds": 2.0,
            "tolerance": 1e-9,
        },
        environment={"pdecert": "0.1.1rc1", "python": "test"},
    )
    manifest_path = directory / "run-manifest.json"
    dump_run_manifest(manifest, manifest_path)
    return manifest_path, manifest


def test_build_dump_and_validate_run_bundle(tmp_path):
    manifest_path, manifest = _write_bundle(tmp_path)

    loaded = validate_run_bundle(manifest_path)

    assert manifest_to_dict(loaded) == manifest_to_dict(manifest)
    assert len(run_manifest_sha256(loaded)) == 64
    assert run_manifest_sha256(loaded) == run_manifest_sha256(manifest)
    with pytest.raises(TypeError):
        loaded.evaluator.configuration["tolerance"] = 1.0


def test_candidate_tampering_breaks_bundle_validation(tmp_path):
    manifest_path, _ = _write_bundle(tmp_path)
    (tmp_path / "candidate.json").write_text('{"fields":{"u":"0"}}\n')

    with pytest.raises(RunManifestError, match="digest mismatch for candidate.json"):
        validate_run_bundle(manifest_path)


def test_report_tampering_breaks_bundle_validation(tmp_path):
    manifest_path, _ = _write_bundle(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(report_path.read_text().replace('"PROVED"', '"REFUTED"'))

    with pytest.raises(RunManifestError, match="digest mismatch for report.json"):
        validate_run_bundle(manifest_path)


def test_builder_rejects_non_strict_json_report(tmp_path):
    _, manifest = _write_bundle(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text('{"residual": NaN}\n')

    with pytest.raises(RunManifestError, match="report is not strict JSON"):
        build_run_manifest(
            bundle_root=tmp_path,
            run_id=manifest.run_id,
            problem_id=manifest.problem_id,
            template_path="template.json",
            candidate_path="candidate.json",
            report_path="report.json",
            artifact_id=manifest.candidate.artifact_id,
            artifact_kind=manifest.candidate.kind,
            field_names=manifest.candidate.field_names,
            provenance=manifest.candidate.provenance,
            evaluator_name=manifest.evaluator.name,
            evaluator_version=manifest.evaluator.version,
            evaluator_configuration=manifest.evaluator.configuration,
            environment=manifest.evaluator.environment,
        )


def test_manifest_and_report_reject_duplicate_json_keys(tmp_path):
    manifest_path, manifest = _write_bundle(tmp_path)
    manifest_path.write_text('{"manifest_version":1,"manifest_version":1}\n')
    with pytest.raises(RunManifestError, match="duplicate object key"):
        load_run_manifest(manifest_path)

    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"PROVED","status":"REFUTED"}\n')
    with pytest.raises(RunManifestError, match="duplicate object key"):
        build_run_manifest(
            bundle_root=tmp_path,
            run_id=manifest.run_id,
            problem_id=manifest.problem_id,
            template_path="template.json",
            candidate_path="candidate.json",
            report_path="report.json",
            artifact_id=manifest.candidate.artifact_id,
            artifact_kind=manifest.candidate.kind,
            field_names=manifest.candidate.field_names,
            provenance=manifest.candidate.provenance,
            evaluator_name=manifest.evaluator.name,
            evaluator_version=manifest.evaluator.version,
            evaluator_configuration=manifest.evaluator.configuration,
            environment=manifest.evaluator.environment,
        )


def test_manifest_rejects_traversal_and_boolean_version(tmp_path):
    _, manifest = _write_bundle(tmp_path)
    payload = manifest_to_dict(manifest)
    payload["problem"]["template"]["path"] = "../template.json"
    with pytest.raises(RunManifestError, match="bundle-relative path"):
        manifest_from_dict(payload)

    payload = manifest_to_dict(manifest)
    payload["manifest_version"] = True
    with pytest.raises(RunManifestError, match="manifest_version"):
        manifest_from_dict(payload)


def test_candidate_fields_must_match_template(tmp_path):
    _, manifest = _write_bundle(tmp_path)

    with pytest.raises(RunManifestError, match="field names do not match"):
        build_run_manifest(
            bundle_root=tmp_path,
            run_id=manifest.run_id,
            problem_id=manifest.problem_id,
            template_path="template.json",
            candidate_path="candidate.json",
            report_path="report.json",
            artifact_id=manifest.candidate.artifact_id,
            artifact_kind=manifest.candidate.kind,
            field_names=("v",),
            provenance=manifest.candidate.provenance,
            evaluator_name=manifest.evaluator.name,
            evaluator_version=manifest.evaluator.version,
            evaluator_configuration=manifest.evaluator.configuration,
            environment=manifest.evaluator.environment,
        )


def test_evaluator_configuration_rejects_non_finite_values(tmp_path):
    _, manifest = _write_bundle(tmp_path)

    with pytest.raises(RunManifestError, match="strict JSON"):
        build_run_manifest(
            bundle_root=tmp_path,
            run_id=manifest.run_id,
            problem_id=manifest.problem_id,
            template_path="template.json",
            candidate_path="candidate.json",
            report_path="report.json",
            artifact_id=manifest.candidate.artifact_id,
            artifact_kind=manifest.candidate.kind,
            field_names=manifest.candidate.field_names,
            provenance=manifest.candidate.provenance,
            evaluator_name=manifest.evaluator.name,
            evaluator_version=manifest.evaluator.version,
            evaluator_configuration={"tolerance": float("nan")},
            environment=manifest.evaluator.environment,
        )


def test_builder_rejects_bundle_inputs_outside_root(tmp_path):
    _, manifest = _write_bundle(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-candidate.json"
    outside.write_text("{}\n")

    with pytest.raises(RunManifestError, match="inside"):
        build_run_manifest(
            bundle_root=tmp_path,
            run_id=manifest.run_id,
            problem_id=manifest.problem_id,
            template_path="template.json",
            candidate_path=outside,
            report_path="report.json",
            artifact_id=manifest.candidate.artifact_id,
            artifact_kind=manifest.candidate.kind,
            field_names=manifest.candidate.field_names,
            provenance=manifest.candidate.provenance,
            evaluator_name=manifest.evaluator.name,
            evaluator_version=manifest.evaluator.version,
            evaluator_configuration=manifest.evaluator.configuration,
            environment=manifest.evaluator.environment,
        )
