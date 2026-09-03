# ADR-0010: Typed cross-artifact Atlas records

**Status:** Accepted
**Date:** 2026-09-02
**Decider:** Orestis Oikonomou

## Context

Atlas version 1 preserves symbolic generator output beside a fully bound case.
Its coverage taxonomy names `callable_model`, but its record format cannot bind
a trained model, candidate-free operator template, and model integrity record.
Treating the callable as a symbolic expression would destroy provenance;
putting a checkpoint path in free-form metadata would make identity and
compatibility unverifiable.

The first mixed format must remain installable with PDECert's core dependencies.
It must not import PyTorch, execute a model, treat a content digest as scientific
evidence, or break the existing symbolic intake and blind-review workflow.

## Decision

Introduce Atlas version 2 as a separate typed-bundle contract. Every record has
common metadata:

- a stable record ID and matched-problem ID;
- an explicit artifact type;
- candidate-free problem-template, artifact, and provenance records;
- SHA-256 references to every bundle file; and
- an annotation object kept separate from machine evidence.

The first record version accepts exactly two artifact lanes:

1. `symbolic_expression`: a restricted field-expression artifact plus the
   byte-preserved raw generator output;
2. `callable_model`: the version-1 frozen callable plus a portable version-2
   integrity record.

Validation checks file bytes, the declared problem and artifact IDs, symbolic
field binding, callable coordinate and output order, canonical weight and
configuration digests, and the integrity source-digest inventory. Transport
validation does not require those source files to be present. A repository
reproduction separately runs the full integrity validator against every bound
source file.

Atlas v1 remains supported and unchanged. The first v2 corpus is a small matched
Fisher--KPP preview, not a replacement for the community intake or an immutable
benchmark release.

## Evidence and trust boundary

A valid Atlas v2 record establishes:

- the exact bytes of each bundled file;
- internal consistency between its template, artifact, and integrity claim;
- the declared provenance and annotation structure.

It does not establish:

- that a candidate satisfies the PDE or its conditions;
- that parsed symbolic fields were extracted correctly from the raw response;
- that declared training occurred or produced the weights;
- authorship, trusted execution, or publisher authenticity;
- a formal neural-network certificate or solution-error bound; or
- an independent human label.

Evaluation reports remain separate artifacts under the decision-evidence
contract. Finite autodiff samples may refute a callable but never prove it.

## Options considered

### Extend Atlas v1 in place

Rejected because v1 embeds candidate expressions inside `case.json`. Adding
callable-only files without a version change would give one version two
incompatible meanings and risk breaking the existing review pipeline.

### Store an external checkpoint URL in record metadata

Rejected because mutable URLs, pickle execution, backend-specific classes, and
unvalidated tensor layouts are outside the current security and portability
boundary.

### Introduce a typed Atlas v2 beside v1

Selected because it makes the representation change explicit, permits a clean
candidate-free template boundary, preserves v1 users, and can be validated
offline without the optional autodiff dependency.

## Consequences

- A symbolic model output and a trained PINN can now share one problem identity
  without sharing a correctness decision.
- Contributors get strict valid, tampered, incompatible, and unsupported
  diagnostics before any model is materialized.
- The version-2 contract is intentionally not generic over arbitrary model
  formats; new lanes require their own representation and threat model.
- Digest-bound blind review, guarded annotation import, and versioned per-record
  symbolic/callable evaluation are available. Independent labels,
  adjudication, comparative baselines, and immutable release tooling for mixed
  records remain later benchmark-science slices.
- Duplicated artifact bytes are acceptable in the small preview because they
  make the bundle self-contained; larger artifact transport needs a separate
  content-addressed storage decision.

## Action items

1. [x] Add Atlas v2, record, and symbolic-artifact schemas.
2. [x] Add no-execution validation for symbolic and frozen-callable bundles.
3. [x] Add a digest-bound Fisher--KPP symbolic/PINN matched corpus.
4. [x] Extend independent review and annotation import to typed records.
5. [x] Add versioned per-record evaluation without cross-artifact aggregation.
6. [ ] Define immutable mixed-corpus release and report manifests.
