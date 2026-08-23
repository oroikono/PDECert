# ADR-0001: Plugin-first extension architecture

**Status:** Accepted
**Date:** 2026-08-24
**Decider:** Orestis Oikonomou

## Context

PDECert needs to support new verification techniques, candidate generators,
and benchmark methods without turning its conservative core into a growing set
of hard-coded branches. Extensions must not weaken the rule that finite
sampling can refute but cannot prove a candidate. Existing JSON inputs and
machine-readable reports must remain compatible.

## Decision

Use a small stable core with ordered, explicitly supplied plugin registries.
The first extension contract is the checker registry:

- every checker receives the same immutable `CheckContext`;
- a checker returns partial evidence in a `CheckResult`;
- proof evidence names obligations already defined by the problem;
- refutation requires a concrete `Witness`;
- only the orchestrator derives the final `PROVED`, `REFUTED`, or
  `INCONCLUSIVE` status;
- registries are immutable and reject duplicate names;
- extension failures are isolated behind a checker-specific error;
- built-in checkers use the same public protocol as external checkers.

Registries are passed explicitly to verification. PDECert will not discover and
execute arbitrary installed plugins implicitly.

## Options considered

### Option A: Continue extending the monolithic verifier

| Dimension | Assessment |
|---|---|
| Initial complexity | Low |
| Long-term scalability | Low |
| Contributor isolation | Low |
| Compatibility risk | High |

**Pros:** Minimal short-term refactoring.
**Cons:** Every new method modifies soundness-sensitive orchestration code.

### Option B: Stable core with explicit plugin registries

| Dimension | Assessment |
|---|---|
| Initial complexity | Medium |
| Long-term scalability | High |
| Contributor isolation | High |
| Compatibility risk | Medium |

**Pros:** Reviewable extension points, deterministic execution, and testable
soundness boundaries.
**Cons:** Public protocols require compatibility discipline.

### Option C: Separate repository for every extension

| Dimension | Assessment |
|---|---|
| Initial complexity | High |
| Discoverability | Low |
| Independent ownership | High |
| Community cohesion | Low |

**Pros:** Extensions can release independently.
**Cons:** Fragments schemas, documentation, tests, and user workflows.

## Trade-off analysis

Option B adds a small amount of architecture now in exchange for a clear
soundness boundary. Explicit registries are preferred over automatic entry
point discovery because verification should never change merely because an
unrelated package is installed in the environment.

## Consequences

- Contributors can implement checks without editing the orchestration loop.
- A registry's ordered checker names are part of a reproducible run setup.
- Third-party checkers remain responsible for the validity of evidence they
  return; PDECert validates obligation scope but cannot prove plugin code sound.
- Benchmark-method and generator-adapter registries should follow the same
  explicit, immutable pattern.
- The public extension API is experimental until the next minor release.

## Action items

1. [x] Define the checker context, result, protocol, and registry.
2. [x] Run all built-in verification stages through the registry.
3. [x] Test ordering, immutability, duplicate names, and obligation scope.
4. [ ] Add a benchmark-method registry.
5. [ ] Add generator-adapter contracts.
6. [ ] Publish a compatibility policy before declaring the API stable.
