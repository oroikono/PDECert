# Compare saved Atlas reports

Use this when you have a PDECert Atlas evaluation and a baseline run over the
same records. The comparison puts their native evidence side by side. It does
not rerun either method, resolve disagreements, or calculate accuracy.

## Core-only example

Run these commands from a checkout with PDECert installed. They use the same
symbolic record from `corpus/matched`; PyTorch, model downloads and API keys are
not needed. A fresh directory keeps earlier reports intact.

```bash
comparison_dir="$(mktemp -d)"
pdecert corpus evaluate corpus/matched --record qwen3-fisher-kpp-01 \
  --output "$comparison_dir/evaluation.json"
pdecert corpus baseline corpus/matched --record qwen3-fisher-kpp-01 \
  --method fixed-collocation --output "$comparison_dir/collocation.json"
pdecert corpus baseline corpus/matched --record qwen3-fisher-kpp-01 \
  --method direct-sympy --output "$comparison_dir/sympy.json"
pdecert corpus compare-reports "$comparison_dir/evaluation.json" \
  "$comparison_dir/collocation.json" --atlas corpus/matched \
  --output "$comparison_dir/compare-collocation.json"
pdecert corpus compare-reports "$comparison_dir/evaluation.json" \
  "$comparison_dir/sympy.json" --atlas corpus/matched \
  --output "$comparison_dir/compare-sympy.json"
```

Each comparison has one joined record. Fixed collocation reports an empirical
`pass`; direct SymPy reports `zero` for the represented residuals and conditions.
Neither is promoted into a PDECert decision. Inspect the full PDECert report
under `records[0].pdecert.report` and the baseline result under
`records[0].baseline`. The record's annotation remains pending. This example is
a workflow smoke test, not an independent model audit or evidence of superiority.

Without `--output`, the command prints JSON. Exit code `0` means that the join
succeeded, not that the candidate is correct. Invalid reports or mismatched
identities return `64` with a diagnostic on stderr. An output file must be new;
the command will not overwrite a report, symlink or earlier comparison.
An invalid `--atlas` is an error; the command never drops source binding and
tries the weaker declared-identity comparison instead.

## Python interface

```python
from pdecert import compare_atlas_reports, load_atlas_baseline_report, load_atlas_evaluation

evaluation = load_atlas_evaluation("evaluation.json")
baseline = load_atlas_baseline_report("collocation.json")
comparison = compare_atlas_reports(evaluation, baseline, atlas_path="corpus/matched")
for row in comparison["records"]:
    print(row["record_id"], row["pdecert"]["report"]["status"], row["baseline"]["outcome"])
```

The API accepts an evaluation version 1 and a baseline report version 1 or 2.
Both inputs are validated through their native report contracts. Returned data
is a defensive copy, so changes to it cannot alter either input.

## Binding to the source Atlas

Supply `--atlas PATH` or `atlas_path=PATH` to check the reports against an actual
Atlas v2 directory. The whole Atlas is validated before its canonical digest is
compared with the reports. Its template, artifact and retained provenance must
match the saved identity; merely copying a digest into both reports is not
enough. This uses the existing Atlas loader without importing PyTorch or
materializing a callable.

The reports may select a subset of the Atlas, but every selected record must
exist and match its problem ID, artifact ID and artifact type. Their record sets
must still match each other exactly. Binding also checks:

- every PDECert evidence-event ID refers to a represented residual, condition,
  or, for symbolic records, field/domain obligation;
- each event witness names the obligation identified by that event;
- a saved `PROVED` claim has discharged events for every represented obligation,
  including symbolic domain checks;
- symbolic baseline checks cover every template residual and condition in
  order, with matching IDs, names and source expressions;
- an empirical baseline witness names a represented constraint and its exact
  source expression; and
- a non-unsupported baseline result accepts the source solution semantics.

Refuted or inconclusive evaluations can stop early, so they need not contain an
event for every obligation. A version-1 empirical baseline does not record
per-obligation coverage; source binding cannot recover that missing information.

Bound comparisons use `comparison_version: 2` and
`comparison_scope: source_atlas_identity_and_obligation_scope_only`. The other
fields and native evidence are unchanged. `ATLAS_BOUND_COMPARISON_VERSION` exposes
this version through the Python API. Source paths are not included in the output.
The Atlas digest includes annotations and retained artifact metadata, but not
its review-neutral README or coverage file. Changing bound source content
requires new reports; binding does not rewrite old ones to make them match.

## Identity and evidence contract

The inputs must declare the same Atlas version, name and SHA-256 digest. Their
nonempty record sets must match exactly: missing or extra records are errors,
not a silently dropped intersection. Every joined record must also match its
`problem_id`, `artifact_id` and `artifact_type`. Input order may differ; output
records are sorted by `record_id`.

Without a supplied Atlas, the original version-1 output is unchanged and contains:

- `comparison_scope: declared_identity_and_native_evidence_only` and
  `evidence_policy: per_record_no_aggregation`;
- the shared declared Atlas identity;
- `sources.evaluation`: the evaluation version, policy, options, runtime and
  canonical validated report digest;
- `sources.baseline`: the baseline report version, policy, complete adapter
  metadata/configuration, runtime and canonical validated report digest;
- per-record identities, the full native PDECert report, and the full native
  baseline result without repeated identity fields.

Witnesses, symbolic checks, unsupported reasons, abstentions and incomplete
obligations remain visible. External version-2 adapters may include empirical
and symbolic rows in one report. A callable PDECert result can appear beside an
unsupported baseline row; loading such saved evidence does not require PyTorch.
No method result is mapped to another method's status, and no overall decision,
agreement score or truth label is inferred. Tolerances and sampling designs may
differ; the retained settings make that difference inspectable, not corrected.
Obligations are not cross-matched merely because their names look similar.

Source digests use each native validator's canonical normalized JSON. They
ignore formatting but retain source array order; sorting the comparison does
not rewrite the source documents. These are content identifiers, not signatures
or hashes of raw file bytes.

## What this does not check

Without source binding, matching declarations do not establish that either
report belongs to the actual Atlas or candidate. With binding, content identity
and obligation references are checked, not the truth of the evidence. Neither
mode replays residual calculations, checks witness coordinates or magnitudes,
authenticates execution, or establishes a mathematical certificate. Free-text
certificate details, `exact_checks` and incomplete-reason descriptions are not
independently interpreted. A fabricated report can still supply known IDs and
invented evidence. Keep the original Atlas, artifacts, scripts and environment
for mathematical replay.
Pending labels remain pending; comparative benchmark claims need independently
reviewed outcomes and a fairness protocol beyond this descriptive view.

For each native contract, see [Atlas evaluations](atlas-evaluation.md) and
[baseline report loading](baseline-adapters.md#loading-saved-reports).
