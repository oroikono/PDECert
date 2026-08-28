# ADR-0003: Explicit decision-evidence levels

**Status:** Accepted
**Date:** 2026-08-26
**Decider:** Orestis Oikonomou

## Context

PDECert is expanding from exact symbolic expressions toward callable models,
numerical fields, interval methods, and weak formulations. These backends do not
produce interchangeable evidence:

- an exact symbolic identity can discharge a represented classical obligation;
- interval arithmetic or an a posteriori theorem can provide a rigorous bound
  for a documented supported class;
- floating-point collocation, automatic differentiation, or quadrature can find
  useful violations but does not become a proof merely because many samples
  pass.

The existing `PROVED`, `REFUTED`, and `INCONCLUSIVE` outcomes describe the
decision but not the mathematical strength behind it. That omission becomes
misleading once exact and numerical backends share one report.

OpenMath is relevant to future expression interchange because it separates an
application's private representation from semantic mathematical objects and
published content dictionaries. It does not supply verification semantics or a
proof engine by itself. PDEBench and SciML libraries are relevant integration
targets, but their sampled tensors, geometries, and residual operators likewise
do not establish rigorous certification on their own.

## Decision

Every decisive `Report` carries `decision_evidence`, with one of three levels:

- `EXACT`: exact algebraic or symbolic evidence for all obligations used in the
  decision;
- `RIGOROUS_BOUND`: validated interval, ball, or a posteriori evidence whose
  assumptions and error bound cover all obligations used in the decision;
- `EMPIRICAL`: floating-point evaluation, collocation, autodiff sampling, or
  unbounded quadrature evidence.

An inconclusive report has no decision-evidence level. A passing empirical check
cannot contribute a proved obligation. The checker orchestrator enforces this:
only `EXACT` and `RIGOROUS_BOUND` may appear with `proved_obligations`. Every
witness must also identify its evidence level.

The existing status vocabulary remains for compatibility. In particular, a
floating-point counterexample can still yield `REFUTED`, but the report exposes
that the decision is `EMPIRICAL`. Consumers requiring mathematical guarantees
can therefore accept only decisions with `EXACT` or `RIGOROUS_BOUND` evidence.

Weak-form residual quadrature starts as `EMPIRICAL`. It may be upgraded to
`RIGOROUS_BOUND` only when the implementation includes validated quadrature or
an explicit a posteriori theorem, constants, assumptions, and a machine-readable
bound.

## Options considered

### Option A: Keep one undifferentiated status

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Scientific clarity | Low |
| Backend extensibility | Low |
| Overclaiming risk | High |

**Pros:** No report or plugin changes.

**Cons:** Exact identities, interval bounds, and floating-point samples appear
equally authoritative.

### Option B: Add enforced decision-evidence levels

| Dimension | Assessment |
|---|---|
| Compatibility | Medium |
| Scientific clarity | High |
| Backend extensibility | High |
| Implementation cost | Low |

**Pros:** Makes current limitations machine-readable and gives future rigorous
backends a sound contract.

**Cons:** Extends serialized reports and tightens the experimental checker API.

### Option C: Give every backend unrelated statuses and reports

| Dimension | Assessment |
|---|---|
| Local simplicity | Medium |
| Cross-backend comparison | Low |
| Long-term maintenance | Low |
| Scientific clarity | Medium |

**Pros:** Each backend can use native terminology.

**Cons:** Prevents one evaluation harness from comparing symbolic, callable,
interval, and numerical artifacts.

## Trade-off analysis

Option B preserves a common decision vocabulary while making the guarantee
boundary explicit. It is deliberately smaller than a full proof-object format:
the immediate goal is to prevent unsound evidence promotion before interval and
weak-form backends arrive. Detailed bound objects and per-obligation provenance
are defined by [`ADR-0009`](0009-versioned-evidence-reports.md).

Adopting OpenMath immediately would address only part of representation
interchange and would not solve branch cuts, weak semantics, resource limits, or
numerical certification. An OpenMath phrasebook should therefore be a separate
adapter decision after the internal PDE operator model is no longer limited to
classical rectangular-domain expressions.

## Consequences

- `PROVED` cannot be produced by empirical evidence, including third-party
  checkers.
- Numeric and autodiff refutations remain useful but are visibly empirical.
- Exact symbolic incompleteness remains an abstention, not a false invalid
  decision.
- Future Arb/FLINT and a posteriori checkers have a defined route to rigorous
  decisions without pretending they are exact identities.
- Existing consumers that require an exact report-key set must update for the
  added `decision_evidence` field.
- Detailed per-obligation evidence is serialized without removing the summary
  fields introduced by this decision.

## Action items

1. [x] Add typed decision-evidence levels to reports and checker results.
2. [x] Reject empirical proof claims and unclassified witnesses.
3. [x] Classify built-in symbolic, floating-point, and autodiff decisions.
4. [x] Add a regression case where SymPy abstains on a domain-valid identity.
5. [ ] Define a versioned rigorous-bound payload with assumptions and constants.
6. [ ] Implement an optional Arb/FLINT interval backend for a restricted class.
7. [ ] Add weak-residual diagnostics without promoting sampled passes to proof.
8. [ ] Design OpenMath and SciML adapters after the internal operator semantics
   support the required boundary and weak-form constructs.

## References

- [OpenMath Standard](https://openmath.org/standard/)
- [OpenMath technical overview](https://openmath.org/technical/)
- [PDEBench repository](https://github.com/pdebench/PDEBench)
- [DeepXDE PDE data API](https://deepxde.readthedocs.io/en/stable/modules/deepxde.data.html)
- [python-flint Arb documentation](https://python-flint.readthedocs.io/en/stable/arb.html)
