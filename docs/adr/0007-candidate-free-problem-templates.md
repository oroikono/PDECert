# ADR-0007: Candidate-free problem templates

**Status:** Accepted
**Date:** 2026-08-28
**Decider:** Orestis Oikonomou

## Context

Schema-v3 cases correctly bind named symbolic candidate expressions to trusted
PDE residuals and conditions, but that makes a saved case fully instantiated.
A benchmark task, agent prompt, symbolic generator, and trained callable need a
candidate-independent identity so that each artifact is evaluated against the
same mathematics. Copying the task into every artifact lane creates semantic
drift; letting a candidate carry its own task lets it accidentally or
deliberately weaken the obligations.

The new boundary must preserve all existing version 1, 2, and 3 case files. It
must also preserve PDECert's evidence contract: sharing a problem definition
does not transfer proof between symbolic and empirical evaluations.

## Decision

Add a separately versioned `ProblemTemplate` representation with:

- a stable name and explicit `classical_strong` solution semantics;
- declared variables, rectangular domains, and parameter assumptions;
- named candidate field slots, with no candidate values;
- named PDE residuals and conditions in the existing restricted operator
  grammar;
- deterministic JSON load, dump, schema, and CLI validation;
- an explicit symbolic binding operation requiring exactly the declared field
  set;
- direct lowering to the existing callable autodiff problem.

Template version 1 accepts only classical strong, pointwise semantics. Every
declared field must appear in a trusted operator. Symbolic binding produces an
ordinary schema-v3 `VerificationCase`, so existing verification code remains
the authority for expression parsing and exact checks. Callable lowering still
has its documented narrower scope and still treats sampled success as
inconclusive.

## Options considered

### Option A: Keep only fully instantiated cases

| Benefit | Cost |
| --- | --- |
| No new public representation | Every artifact duplicates the trusted task |
| Existing API remains sufficient for symbolic checks | Cross-artifact identity is informal and drift-prone |
| Candidate and obligations travel in one file | A generator can appear to define the task it is judged against |

### Option B: Make candidate fields optional in schema v4

| Benefit | Cost |
| --- | --- |
| One JSON family for templates and cases | File meaning depends on whether a member is absent |
| Fewer top-level types | Existing code must handle partially instantiated cases |
| A direct schema evolution path | Migration risk without improving evidence semantics |

### Option C: Add a separate template format and explicit binding

| Benefit | Cost |
| --- | --- |
| Problem ownership and candidate ownership are explicit | One additional versioned format |
| Existing case schemas remain unchanged | Conversion and validation APIs must be maintained |
| Symbolic and callable lanes share one task identity | Version 1 deliberately covers only classical strong problems |

Option C is selected because the lifecycle distinction is real and should not
be encoded as a nullable field.

## Trade-off analysis

| Criterion | Fully instantiated only | Optional schema-v4 fields | Separate template |
| --- | ---: | ---: | ---: |
| Backward compatibility | High | Medium | High |
| Problem/candidate separation | Low | Medium | High |
| Serialization clarity | Medium | Low | High |
| Initial implementation cost | Low | Medium | Medium |
| Cross-artifact reuse | Low | Medium | High |
| Risk of evidence conflation | Medium | Medium | Low |

The selected design adds a small amount of API surface in exchange for a
clearer trust boundary and no migration of public case data.

## Consequences

- Benchmark and agent tasks can publish the mathematics before any candidate is
  generated.
- Symbolic and callable artifacts can target one template without sharing their
  evidence or status.
- Existing case readers, writers, corpus records, and releases are unchanged.
- The template parser reuses case-schema semantic validation instead of
  introducing a second expression language.
- Weak forms, entropy conditions, irregular geometry, and parameterized
  callable evaluation remain unsupported and must not be encoded as classical
  pointwise tasks.
- A future backend-neutral task runtime can use templates as input while
  keeping candidate provenance and replay manifests separate.

## Action items

1. [x] Add the version-1 Python type, parser, deterministic writer, and JSON schema.
2. [x] Add exact-field symbolic binding and case-to-template conversion.
3. [x] Compile templates directly into the callable autodiff representation.
4. [x] Add CLI validation, valid/invalid/unsupported tests, and a two-lane example.
5. [ ] Add stable template identifiers and digest-bound run manifests.
6. [ ] Exercise the contract through one provider-neutral agent task adapter.
7. [ ] Evaluate a later semantics version only after a backend can faithfully
   represent and check it.
