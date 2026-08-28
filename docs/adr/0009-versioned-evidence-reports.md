# ADR-0009: Versioned obligation-level evidence reports

**Status:** Accepted
**Date:** 2026-08-28
**Decider:** Orestis Oikonomou

## Context

The original `Report` exposes a conservative decision, one summary evidence
level, exact-check labels, incomplete reasons, and at most one refutation
witness. That interface prevents the most important overclaim—sampled success
cannot become `PROVED`—but it loses how individual obligations were treated.

Cross-artifact evaluation needs to distinguish, for example, an exact PDE
identity from an empirically passing boundary sample. Rigorous backends also
need to state which quantity they bound. A uniform residual bound, a boundary
trace bound, and a solution-error guarantee are not interchangeable.

The contract must remain useful to current CLI, benchmark, agent, and plugin
consumers. This decision therefore adds a versioned evidence stream without
renaming the existing Python `Report` or removing its summary fields.

## Decision

Report JSON version 1 adds:

- `report_version`, which versions the serialized representation;
- `aggregation_policy_version`, which versions how partial checker evidence is
  combined into a decision; and
- `evidence_events`, an ordered list of checker-attributed events bound to
  stable obligation identifiers.

Each `EvidenceEvent` records an evidence kind, outcome, strength, explanation,
and optional witness or rigorous-bound payload. Version 1 supports:

- exact certificates that discharge or refute an obligation;
- rigorous bounds that discharge an obligation and name the bounded quantity,
  norm, scope, assumptions, and constants;
- empirical counterexamples with replayable witnesses;
- empirical passes that remain non-decisive; and
- abstentions with a reason.

`BoundEvidence.bound_type` distinguishes uniform residual, boundary trace,
solution error, and explicitly named other quantities. This prevents a residual
enclosure from being presented as a solution-error guarantee.

The existing fields—`status`, `decision_evidence`, `exact_checks`,
`incomplete_reasons`, `witness`, and `max_sampled_residual`—remain in version 1.
They are compatibility summaries. New consumers should inspect
`evidence_events`. Existing exact third-party checkers that use the documented
`CheckResult` fields receive synthesized exact events. A checker claiming a
rigorous bound must provide the structured event; a bare
`proof_level=RIGOROUS_BOUND` is rejected.

Non-finite measurements are represented as the string `"infinity"`, rather
than non-standard JSON constants. The canonical schema is
[`schema/report-v1.schema.json`](../../schema/report-v1.schema.json).

## Options considered

### Option A: Keep summary-only reports

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Cross-artifact auditability | Low |
| Bound semantics | Low |
| Implementation complexity | Low |

**Pros:** No serialized changes.

**Cons:** Consumers cannot determine which obligation received which evidence,
and rigorous guarantees remain underspecified.

### Option B: Add versioned events while retaining summaries

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Cross-artifact auditability | High |
| Bound semantics | High |
| Implementation complexity | Medium |

**Pros:** Establishes one inspectable evidence contract while preserving the
current Python object and common report keys.

**Cons:** Reports grow, plugins must eventually emit richer events directly,
and two representations of summary information must remain consistent.

### Option C: Replace `Report` with backend-specific certificate objects

| Dimension | Assessment |
|---|---|
| Compatibility | Low |
| Cross-artifact auditability | Low |
| Local backend flexibility | High |
| Implementation complexity | High |

**Pros:** Every backend can expose its native representation.

**Cons:** Removes the common evaluation layer and forces every downstream
consumer to understand every backend.

## Trade-off analysis

Option B owns the narrow interoperability boundary PDECert needs: common
decision semantics without pretending that evidence is interchangeable. The
version fields make future migration explicit. Retained summaries avoid an
unnecessary break during the alpha period, while structured bound requirements
intentionally reject scientifically incomplete rigorous claims.

The first version does not model proof-assistant kernels, signed execution
attestations, lower bounds used for rigorous refutation, or multi-obligation
joint certificates. Those additions require a new event or report version
rather than reinterpretation of version 1.

## Consequences

- Every built-in checker emits obligation-level events.
- A passing empirical check is visible as `OBSERVED_PASS` but cannot discharge
  an obligation or produce `PROVED`.
- Report JSON is strict and round-trippable through the public loader.
- Bare rigorous-bound declarations from experimental plugins now fail with an
  actionable contract error.
- Existing exact plugins continue to work through an explicitly documented
  compatibility adapter.
- The version-1 schema cannot claim a rigorous refutation; such a checker must
  abstain until that evidence shape is designed.

## Action items

1. [x] Add evidence, outcome, and bound types to the public Python API.
2. [x] Emit per-obligation events from symbolic and callable built-ins.
3. [x] Add deterministic report loading and dumping.
4. [x] Publish the version-1 JSON Schema.
5. [x] Add compatibility, invalid, and unsupported-input tests.
6. [ ] Add immutable problem, candidate, evaluator, and environment references
   through the existing run-manifest layer rather than duplicating them here.
7. [ ] Define a new schema version before representing rigorous refutations or
   joint certificates.
