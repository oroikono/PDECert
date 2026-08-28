# Digest-bound run manifests

A run manifest answers a narrow but important question: **which exact bytes and
configuration produced this evaluation record?** It binds a candidate-free
problem template, one candidate artifact, evaluator settings, environment, and
output report with SHA-256 digests.

This is reproducibility infrastructure, not a new verification backend. A
valid digest confirms content identity only. It does not prove that the report
is mathematically correct, that the evaluator ran in a trusted environment, or
that the named producer authored the artifact.

## Validate the example bundle

The repository includes a complete symbolic heat run:

```bash
pdecert template validate examples/heat-template.json
pdecert run validate examples/heat-run-manifest.json
```

The second command verifies the exact template, candidate, and report bytes,
checks that every path stays inside the manifest directory, loads the template,
and confirms that candidate field names match the task. It prints the canonical
manifest digest and the recorded run identities.

If one character in the candidate or report changes, validation fails with the
expected and actual file digest. Reports must be strict JSON objects; `NaN` and
other non-standard constants and duplicate object keys are rejected.

## Build a manifest

All inputs must already exist inside one bundle root:

```python
from pdecert import build_run_manifest, dump_run_manifest

manifest = build_run_manifest(
    bundle_root="examples",
    run_id="heat-exact-symbolic-01",
    problem_id="heat-classical-01",
    template_path="heat-template.json",
    candidate_path="exact-heat-candidate.json",
    report_path="exact-heat-report.json",
    artifact_id="exact-heat-expression-01",
    artifact_kind="symbolic",
    field_names=("u",),
    provenance={"producer": "PDECert maintainers", "revision": "examples-v1"},
    evaluator_name="pdecert.symbolic",
    evaluator_version="0.1.1rc1",
    evaluator_configuration={
        "max_expression_ops": 10_000,
        "samples_per_axis": 5,
        "symbolic_timeout_seconds": 2.0,
        "tolerance": 1e-9,
    },
    environment={"pdecert": "0.1.1rc1", "python": "3.12", "sympy": "1.14.0"},
)
dump_run_manifest(manifest, "examples/heat-run-manifest.json")
```

The candidate file is intentionally opaque at this layer. It may be a symbolic
field mapping, frozen model weights and architecture metadata, an unedited LLM
response, or generated source code. Its representation-specific loader remains
responsible for interpreting it safely.

## Scope and unsupported claims

Version 1 supports local, regular files inside one manifest directory and
SHA-256 content identity. It does not yet provide:

- digital signatures or publisher authenticity;
- proof that an evaluator was actually executed;
- container, hardware, or operating-system reproduction;
- remote URL fetching or large-file storage;
- semantics for parsing every candidate artifact kind;
- any upgrade from empirical diagnostics to formal proof.

The canonical format is
[`schema/run-manifest-v1.schema.json`](../schema/run-manifest-v1.schema.json),
and the threat-model decision is recorded in
[`ADR-0008`](adr/0008-digest-bound-run-manifests.md).
