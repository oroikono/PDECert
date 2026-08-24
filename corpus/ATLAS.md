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

The version 1 corpus currently serializes symbolic expressions from open models
and symbolic solvers. The repository can already check PyTorch callables in
memory, but it does not yet define a portable corpus representation for model
weights and executable residual operators. Numerical fields and generated
solver programs likewise need explicit formats and security boundaries before
they enter a release.

Until those formats exist:

- submit symbolic records as corpus data;
- submit callable, PINN, neural-operator, numerical, or generated-program cases
  through the failure-case issue form;
- do not convert an unsupported artifact into a symbolic expression and present
  it as the original output.

## Modular record bundles

The community atlas stores one contribution per directory so unrelated pull
requests do not edit the same large JSON document:

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

## Contributing a case

Start with the
[failure-case issue form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml).
Maintainers can help reduce a large artifact to the smallest case that preserves
the failure.

When contributing a versioned corpus file, validate it locally:

```bash
pdecert corpus validate path/to/corpus.json
```

The command validates every embedded case, origin record, raw-output digest,
annotation state, and record identifier. It then prints a compact coverage
summary. Validation does not establish that a human label is correct.

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
