# PDE Failure Atlas

The PDE Failure Atlas is a community corpus of solution artifacts that expose a
specific verification difficulty. Each record binds a candidate to the problem
it claims to solve, its unedited source output, provenance, a human annotation,
and machine-checkable evidence where the current verifier can provide it.

The atlas is not a leaderboard of PDE solvers. Its purpose is to make failure
modes reproducible and to measure whether a checker proves, refutes, or abstains
for the right reason.

## What makes a useful record

A proposed record should contain:

- a fully stated PDE problem, including domains, parameters, initial or boundary
  conditions, and intended solution semantics;
- the candidate artifact exactly as produced, without silently repairing it;
- enough origin information to reproduce or identify the generating system;
- a concrete reason the case is interesting, such as a missed boundary failure,
  a singularity between sample points, or an unsupported solution notion;
- licensing or terms information sufficient for redistribution review.

Both valid and invalid candidates are useful. Valid records test false
rejections and proof coverage. Invalid records test whether a verifier returns a
concrete witness. Underspecified cases may be labeled `unclear`; they should not
be forced into a binary label.

Synthetic mutations are allowed when they isolate one mechanism, but they must
be identified as synthetic. Naturally occurring outputs from solvers, trained
models, and scientific agents are the priority because they better represent
real use.

## Current and planned artifact lanes

The version 1 community corpus serializes symbolic expressions from open
models, symbolic solvers, and explicitly identified synthetic constructions.
Atlas version 2 adds typed bundles that bind a candidate-free problem template
to either a symbolic expression artifact or the portable, non-executing frozen
callable format. The initial [`corpus/matched`](matched/README.md) preview holds
one Qwen3/Fisher--KPP symbolic artifact and one separately trained PINN under the
same problem ID. Core validation does not import PyTorch or execute either
artifact.

Version 2 is an intake, representation, and guarded independent-review contract.
Its mixed records remain `pending`, and the baseline and immutable-release
pipeline has not yet been generalized to callable models. Numerical fields and
generated solver programs likewise need explicit formats and security
boundaries before they enter a release.

Until those formats exist:

- submit symbolic records as corpus data;
- submit callable and PINN cases through the failure-case issue form so a
  maintainer can determine whether the restricted frozen format and Atlas v2
  record apply;
- submit neural-operator, numerical, or generated-program cases through the same
  issue form until their representation contracts exist;
- do not convert an unsupported artifact into a symbolic expression and present
  it as the original output.

## Modular record bundles

Both Atlas versions store one contribution per directory so unrelated pull
requests do not edit the same large JSON document. Version 1 uses:

~~~text
community-atlas/
├── atlas.json
└── records/
    └── <record-id>/
        ├── record.json
        ├── case.json
        └── raw-output.txt
~~~

The atlas manifest supplies the collection name, description, and format
version. Record metadata stays separate from the latest problem-schema case and
the byte-preserved raw output. Directory names must match record IDs, and loose
files or symlinked record directories are rejected.

The monolithic corpus format remains supported for immutable releases and
backward compatibility.

Version 2 separates the problem from the candidate and digest-binds each file:

~~~text
matched-atlas/
├── atlas.json
├── coverage.json
└── records/
    └── <record-id>/
        ├── record.json
        ├── template.json
        ├── artifact.json
        ├── raw-output.txt  # symbolic records only
        └── integrity.json  # frozen callables only
~~~

The shared `problem_id` groups unlike artifacts without combining their
reports. The symbolic artifact records parsed fields beside the exact raw-
output digest; validation does not prove that the extraction was semantically
correct. A callable record binds the restricted frozen model to its
configuration, weights, and source-digest inventory. File digests establish
content identity only; they do not prove provenance claims or PDE correctness.
The public contracts are
[`atlas-v2.schema.json`](../schema/atlas-v2.schema.json),
[`atlas-v2-record-v1.schema.json`](../schema/atlas-v2-record-v1.schema.json),
[`atlas-review-v2.schema.json`](../schema/atlas-review-v2.schema.json), and
[`symbolic-artifact-v1.schema.json`](../schema/symbolic-artifact-v1.schema.json).
See [ADR-0010](../docs/adr/0010-typed-cross-artifact-atlas-records.md).

## Record lifecycle

1. **Proposed:** a contributor opens a failure-case issue with a complete
   problem and unedited artifact.
2. **Pending:** provenance and schema checks pass, but no label is exposed to a
   reviewer.
3. **Labeled:** one person completes the blind protocol in `LABELING.md`.
4. **Adjudicated:** a second reviewer resolves a disagreement or ambiguous case.
5. **Released:** the record is included in a digest-bound corpus and benchmark
   report.

PDECert output is comparison evidence, not the human ground truth. A sampled
pass is never promoted to a proof.

Atlas v2 review files are bound to a canonical digest of the exact loaded Atlas.
That digest covers the manifest fields and loaded records, including their bound
problem and artifact contents; review-neutral README and coverage bytes are
excluded. The importer still preserves those files byte-for-byte.
Their decision basis is artifact-aware: symbolic records may use a direct manual
derivation; an invalid callable needs an independent counterexample or rigorous
external certificate; a valid callable needs a rigorous external certificate;
and an undecidable scope should remain `unclear`. Import creates a new Atlas,
adds the declared review basis to each completed annotation, and preserves every
problem, artifact, raw-output, integrity, coverage, and README byte.

## Contributing a case

Start with the
[failure-case issue form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml).
Maintainers can help reduce a large artifact to the smallest case that preserves
the failure.

When contributing a versioned corpus file, validate it locally:

```bash
pdecert corpus validate path/to/corpus.json
pdecert corpus validate path/to/modular-atlas
pdecert corpus validate corpus/matched
```

For version 1, the command validates every embedded case, origin record,
raw-output digest, annotation state, and record identifier. For version 2, it
also validates template/artifact compatibility, every bundle digest, and the
transported frozen-callable integrity claim. It then prints a compact coverage
summary. Validation neither evaluates the PDE nor establishes that a human
label is correct.

## Explicit coverage taxonomy

A modular Atlas may include `coverage.json`, validated against
[`atlas-coverage-v1.schema.json`](../schema/atlas-coverage-v1.schema.json). The
file must contain exactly one entry for every record: missing and unknown record
IDs are rejected. Each entry declares:

- `artifact_type`: `symbolic_expression`, `callable_model`, `numerical_field`,
  or `solver_program`;
- `pde_families`: one or more lowercase taxonomy slugs;
- `spatial_dimension`: the number of spatial coordinates, excluding time and
  parameters.

PDE-family slugs are intentionally extensible rather than a closed enum. Use an
established slug when one exists, and document a genuinely new family in the
same contribution. The taxonomy describes coverage; it is not a correctness
label and is not shown during blind review.

When `coverage.json` is present, `pdecert corpus validate` reports counts for
artifact types, PDE families, and spatial dimensions after checking that the
taxonomy and record set agree.

## Coverage plan

Growth is tracked across independent axes rather than by record count alone:

- PDE family and differential order;
- linear, nonlinear, coupled, and parameterized systems;
- classical, weak, and other explicitly stated solution semantics;
- symbolic, callable, gridded, and program artifacts;
- generator family and model or solver revision;
- regularity, domain, residual, initial, boundary, parameter, extraction, and
  semantics failures;
- cases where collocation, direct simplification, and PDECert disagree.

The first community release targets at least 100 records, but that number is a
coverage milestone rather than a claim of statistical representativeness. A
release report must publish the complete coverage table and known blind spots.

## Review principles

- Prefer one minimal, reproducible failure over a large opaque run.
- Preserve raw evidence and distinguish observation from interpretation.
- Credit submitters and reviewers publicly unless they request otherwise.
- Treat licensing uncertainty as a release blocker, not a footnote.
- Keep unsupported mathematics `INCONCLUSIVE` until a sound checker exists.
- Never claim novelty, correctness, or generality from record volume alone.
