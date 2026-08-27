# ADR-0004: Community evolution through vertical slices and release gates

**Status:** Proposed
**Date:** 2026-08-26
**Decider:** Orestis Oikonomou

## Context

PDECert is expanding from a symbolic pilot into a cross-artifact verification
and evaluation package. The adjacent research landscape is already mature:
SciML benchmarks, PINN libraries, rigorous neural verification, validated
numerics, formal checking, and PDE specification languages each solve parts of
the problem.

The project therefore cannot grow coherently through unrelated feature additions
or record-count targets. It needs a development model that lets contributors add
one artifact type, backend, benchmark case, or integration without weakening
evidence semantics or modifying unrelated subsystems. Recurring development must
also produce reviewable scientific progress rather than activity-only commits.

## Decision

Organize public work into six contributor workstreams:

1. soundness core;
2. artifact representations;
3. verification backends;
4. benchmark science;
5. ecosystem integrations; and
6. developer experience.

Require contributions to be vertical slices. A complete slice includes a stated
mathematical or user-facing scope, implementation behind an explicit boundary,
valid/invalid/unsupported tests, documentation, and a reproducible example or
fixture.

Use evidence-based release gates rather than commit, issue, pull-request, or
record counts. Recurring work selects the next unmet gate, checks for overlapping
open work, and ships at most one coherent pull request per run.

Keep the architecture plugin-first and dependency-light:

- extensions are explicitly registered;
- artifact-specific semantics remain in representation backends;
- optional integrations do not become core dependencies by default;
- machine evidence, human labels, and candidate provenance remain separate;
- unsupported mathematics remains inconclusive.

## Options considered

### Option A: Feature queue organized by module

| Dimension | Assessment |
| --- | --- |
| Initial simplicity | High |
| Scientific coherence | Low |
| Contributor discoverability | Medium |
| Soundness risk | High |

**Pros:** Easy to create a list of files and features.

**Cons:** Encourages local changes that do not deliver a reproducible capability
and makes research claims emerge accidentally from implementation details.

### Option B: Corpus growth as the primary goal

| Dimension | Assessment |
| --- | --- |
| Measurability | High |
| Benchmark breadth | Medium |
| Methodological depth | Low |
| Risk of vanity progress | High |

**Pros:** Record counts are visible and easy to communicate.

**Cons:** More easy candidates do not establish cross-artifact usefulness,
verifier calibration, rigorous evidence, or community adoption.

### Option C: Workstreams, vertical slices, and release gates

| Dimension | Assessment |
| --- | --- |
| Initial documentation cost | Medium |
| Scientific coherence | High |
| Contributor isolation | High |
| Reviewability | High |

**Pros:** Each change has an observable outcome, appropriate tests, and a clear
owner boundary. Multiple contributors can work in parallel without redefining
the core.

**Cons:** Some desirable features remain blocked until representation and
evidence contracts are defined.

## Trade-off analysis

Option C creates more design work before implementation, but PDECert is a
soundness-sensitive package. A fast feature that blurs empirical residuals with
proof would damage the central research claim. Vertical slices also reduce the
cost of external review because code, evidence, examples, and limitations arrive
together.

The workstream model does not require separate teams or packages today. It is an
ownership and dependency model that can later support dedicated maintainers or
optional distributions if adoption warrants them.

## Consequences

- Contributors can locate a change by responsibility and reuse a common
  definition of done.
- The public roadmap can express outcomes without becoming a private task list.
- Daily development reports progress against scientific release evidence rather
  than GitHub activity.
- Cross-cutting changes to statuses, aggregation, or schemas require a higher
  review burden.
- Large backends must be decomposed into representation, validation, checking,
  and benchmark slices.
- Some work may correctly end with documentation or an inconclusive result when
  the proposed mathematics is not yet defensible.

## Action items

1. [x] Publish the package-layer and contributor-workstream map.
2. [x] Add a shared vertical-slice definition of done.
3. [x] Reframe the roadmap around the first publishable cross-artifact release.
4. [ ] Stabilize explicit decision-evidence levels.
5. [ ] Add a matched-case contract spanning symbolic and callable artifacts.
6. [ ] Add a gridded numerical artifact before claiming neural-operator support.
7. [ ] Reproduce or integrate one rigorous external verification backend.
8. [ ] Publish compatibility policy for third-party backends.

