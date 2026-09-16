# Benchmark coverage audit

The coverage audit recounts stored annotations before a benchmark result is
interpreted. It exposes the pilot's origin/verdict separation and the recorded
reviewer coverage. It does not calculate accuracy or establish independent
ground truth.

From a source checkout with PDECert's core dependencies installed:

```bash
python -m experiments.audit_benchmark_coverage
python -m experiments.audit_benchmark_coverage corpus/community
python -m experiments.audit_benchmark_coverage corpus/matched
```

The default input is `corpus/pilot.json`. Each invocation prints deterministic
JSON to standard output; the command writes no files. Exit code `0` means the
source was loaded and counted, not that any candidate or label is correct.
Unreadable, malformed, and unsupported input produces an error on standard
error and exit code `2`, without a partial report.

## Supported scope

The command uses `load_corpus_source` to validate a version-1 monolithic corpus,
a modular Atlas v1, or a supported typed Atlas v2. It uses the existing canonical
corpus/Atlas digest to bind counts to the loaded records, including annotations
and provenance. This is content identity, not a proof or source-history audit.
Atlas documentation and optional `coverage.json` taxonomy are outside the loaded
record digest; taxonomy entries are not substituted for annotation verdicts.

The counting unit is one candidate record. The report includes:

- `origin_verdict_counts`: counts for `valid`, `invalid`, `unclear`, and
  `pending`, grouped explicitly by `origin.kind`, with source record IDs;
- `producer_identifiers`: declared producer/model identifiers within each
  origin kind, with counts;
- `annotation_coverage`: counts for `pending`, `labeled`, and `adjudicated`,
  plus completed and pending denominators;
- `reviewer_coverage`: distinct annotator-ID count, records with zero, one, or
  multiple IDs, and the record IDs attributed to each annotator; and
- `origin_verdict_diagnostic`: the binary-label denominator and whether each
  observed origin has only one binary verdict.

`origin_determines_binary_verdict` is `true` only when at least two origin kinds
and both binary verdicts occur, and every origin has a single observed binary
verdict. It is `false` when that comparison is possible but an origin contains
both verdicts. It is `null` when there are too few origins or verdicts. Pending
and unclear records are excluded only from this diagnostic, with their excluded
count retained. A `false` result is not evidence that a corpus is unconfounded.

Origin kinds are coarse categories. Declared producer/model identifiers do not
establish generator independence or representative diversity. The audit does
not compare evaluator outputs, run inference or training, obtain labels, or
execute callable candidates. Loading symbolic records does parse their declared
problem representation through the existing loader.

## Existing corpus counts

Recount of the committed records used by this change:

| Source | Origin kind | Valid | Invalid | Unclear | Pending |
| --- | --- | ---: | ---: | ---: | ---: |
| `corpus/pilot.json` | `symbolic_solver` | 10 | 0 | 0 | 0 |
| `corpus/pilot.json` | `open_model` | 0 | 10 | 0 | 0 |
| `corpus/community` | `open_model` | 0 | 0 | 0 | 7 |
| `corpus/community` | `synthetic` | 0 | 0 | 0 | 6 |
| `corpus/matched` | `open_model` | 0 | 0 | 0 | 1 |
| `corpus/matched` | `trained_model` | 0 | 0 | 0 | 1 |

The pilot's loaded-corpus digest is
`4be9178edd30fcc561f21e83375713f4b38338484d75a0c8a7c8088e9c4369fb`.
All 20 records are `labeled` with one public annotator ID, `oroikono`; none is
`adjudicated` or lists a second ID. The declared generators are
`sympy.solvers.pde.pdsolve` and `mlx-community/Qwen3-0.6B-4bit`.

This pilot perfectly separates origin kind and stored binary verdict: every
solver output is labeled valid and every open-model output invalid. These
records cannot disentangle discrimination of validity from origin-associated
differences. They do not support an estimate of broad verifier or model accuracy.
Within-origin valid and invalid cases are an explicit missing comparison.

The community digest is
`d295272e238c9134e8173f556a951746132096856d5410cfa4a92b8a81458b6c`;
the matched Atlas v2 digest is
`9cfcf50974491dfc2b1e0a681e0dff11ddeb8420276d48e291c8623e8168a86f`.
Both have zero completed annotations and zero recorded annotator IDs. Pending
records are not invalid, unclear, or independently labeled examples. The
Fisher--KPP open-model record appears in both corpora, so their record counts
must not be added as independent observations. Two matched artifacts also do
not constitute two independent mathematical problems.

## Review evidence still needed

Annotator IDs are metadata. Even multiple IDs or `adjudicated` status do not
establish distinct people, blindness to machine outputs, external affiliation,
or independent mathematical review. The report therefore always leaves
`independent_reviewer_count` as `null` and marks independence as
`not_established_by_annotation_metadata`. This does not assert that a review did
not occur; it identifies what this recount cannot establish.

The pilot annotations contain no second recorded label set. Community and
matched records still require completed review. Evidence for independent labels
and disagreement resolution must come from the
[labeling protocol](../corpus/LABELING.md), retained review provenance, and
actual reviewers. Machine outputs and agreement with the pilot labels cannot
fill that gap. The public-audit roadmap gate remains incomplete.

No frozen corpus, label, benchmark result, or training artifact is rewritten by
this audit. See the [limitations statement](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md)
for the wider interpretation of benchmark evidence.
