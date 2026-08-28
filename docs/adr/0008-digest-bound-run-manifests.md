# ADR-0008: Digest-bound evaluation run manifests

**Status:** Accepted
**Date:** 2026-08-28
**Decider:** Orestis Oikonomou

## Context

Problem templates separate trusted mathematics from candidate artifacts, but a
benchmark or agent result is still ambiguous unless it records exactly which
template, candidate bytes, evaluator configuration, environment, and report
were used. Filenames, model aliases, and mutable URLs are insufficient for
reproduction. Callable weights and generated programs also cannot be safely or
portably reconstructed from an in-memory Python object.

The contract must work for symbolic, callable, and program artifacts without
pretending that a cryptographic digest proves a PDE claim. Version 1 must remain
lightweight, offline-validatable, and free of a signing service or registry.

## Decision

Add a versioned `RunManifest` that binds, by SHA-256:

- one candidate-free `ProblemTemplate` file and a stable problem identifier;
- one opaque candidate artifact file, its kind, declared fields, identifier,
  and string-valued provenance;
- one evaluator name, version, strict-JSON configuration, and runtime
  environment;
- one strict-JSON output report.

All referenced paths are normalized, bundle-relative paths. Validation rejects
path traversal, files outside the manifest directory, missing files, digest
mismatches, invalid templates, non-JSON reports, and candidate field sets that
do not match the template.

Every manifest declares `integrity_scope: content_identity_only`. A valid
manifest demonstrates that the referenced bytes match the record. It does not
establish authorship, trusted execution, mathematical correctness, or the
evidence strength of the report.

## Options considered

### Option A: Record provenance as unvalidated metadata in each report

| Dimension | Assessment |
| --- | --- |
| Complexity | Low |
| Artifact coverage | Medium |
| Tamper detection | Low |
| Offline reproduction | Low |

**Pros:** minimal new code and no separate files.

**Cons:** report producers can omit or rename fields; template, weights, raw
agent output, and report identity remain informal.

### Option B: Use an unsigned, content-addressed bundle manifest

| Dimension | Assessment |
| --- | --- |
| Complexity | Medium |
| Artifact coverage | High |
| Tamper detection | High |
| Offline reproduction | High |

**Pros:** backend-neutral, deterministic, easy to archive, and independently
checkable with standard SHA-256 tooling.

**Cons:** content identity is not publisher authenticity, and large artifacts
still require external distribution conventions.

### Option C: Require signed in-toto or SLSA attestations immediately

| Dimension | Assessment |
| --- | --- |
| Complexity | High |
| Artifact coverage | High |
| Tamper detection | High |
| Offline reproduction | Medium |

**Pros:** stronger supply-chain and publisher-identity story.

**Cons:** key management, CI identity, verification policy, and external tooling
would dominate the first interoperability slice.

## Trade-off analysis

| Criterion | Report metadata | Digest manifest | Signed attestation |
| --- | ---: | ---: | ---: |
| Deterministic identity | Low | High | High |
| Local usability | High | High | Medium |
| Backend neutrality | Medium | High | High |
| Operational burden | Low | Low | High |
| Publisher authenticity | None | None | High |
| Appropriate for current alpha | Low | High | Low |

Option B is selected. It establishes the missing reproducibility boundary while
leaving authenticity and trusted-execution claims for a later threat model.

## Consequences

- Symbolic expressions, model-weight files, raw agent outputs, and generated
  programs can use the same run-level identity contract.
- A reviewer can detect any change to the problem, candidate, or report before
  comparing results.
- Candidate content remains opaque to the manifest layer; representation
  backends retain responsibility for parsing and execution.
- Evaluator configuration is immutable strict JSON, so `NaN`, executable
  objects, and environment-dependent Python values are rejected.
- Manifests do not prove that a run occurred, that code was trustworthy, or
  that a `PROVED` result is sound.
- Remote artifact stores, signed attestations, and large-file transport remain
  later integrations.

## Action items

1. [x] Add immutable Python records, deterministic serialization, and a JSON schema.
2. [x] Validate bundle-local paths, digests, templates, candidate fields, and reports.
3. [x] Add CLI validation, tamper tests, and one complete symbolic run bundle.
4. [ ] Add a provider-neutral agent runner that emits manifests for raw model outputs.
5. [ ] Add callable-model examples that bind frozen weights and architecture configuration.
6. [ ] Evaluate signed attestations after defining publisher and CI trust policies.
