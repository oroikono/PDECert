# ADR-0002: General solution artifacts and conservative backend verification

**Status:** Accepted
**Date:** 2026-08-24
**Decider:** Orestis Oikonomou

## Context

PDECert's first release verifies analytical SymPy expressions whose PDE and
condition residuals are already materialized as symbolic expressions. That is a
useful soundness-sensitive core, but it excludes common scientific-ML outputs:
PINNs, neural operators, numerical fields, and generated solver programs.

These representations do not share one evaluation model. A symbolic expression
can sometimes discharge a global identity, while a PyTorch module provides
automatic derivatives only at evaluated points. A numerical trajectory has no
automatic derivative at all, and executing a generated program introduces a
security boundary. Treating all of these as if they had equal proof strength
would make reports misleading.

The extension must preserve existing Python and JSON behavior. In particular,
finite samples may refute an artifact with a reproducible witness but may not
produce `PROVED`.

## Decision

Introduce a small `SolutionArtifact` protocol with stable field names and an
explicit representation kind. Provide two initial implementations:

- `SymbolicCandidate` stores one or more named SymPy expressions;
- `CallableCandidate` stores differentiable named fields and an explicit
  autodiff backend.

Keep representation-specific problem definitions and checkers instead of
forcing every representation into a symbolic schema:

- the existing `Problem` and symbolic checker registry remain unchanged;
- `AutodiffProblem` represents residual operators plus initial or boundary
  surfaces through fixed coordinates;
- `AutodiffResidualChecker` evaluates PyTorch fields and their derivatives on
  deterministic points;
- `verify_artifact` rejects incompatible problem/artifact pairs explicitly and
  routes compatible pairs to their backend.

The existing `verify(problem, expressions)` API remains supported. It constructs
a `SymbolicCandidate` internally and produces the same report as before.

Passing automatic-differentiation samples produces `INCONCLUSIVE`, accompanied
by an explicit reason. A residual above tolerance or a non-finite residual
produces `REFUTED` with the concrete evaluation point. Future interval or formal
checkers may prove obligations only for their documented supported class.

PyTorch is an optional dependency. Importing PDECert or using symbolic
verification must not import or require it.

## Options considered

### Option A: Keep a symbolic-only candidate model

| Dimension | Assessment |
|---|---|
| Initial complexity | Low |
| Scientific-ML reach | Low |
| Backward compatibility | High |
| Community extension surface | Low |

**Pros:** No new abstraction or dependency boundary.

**Cons:** PINNs and neural operators remain outside the project, and generated
artifacts cannot share reporting or checker infrastructure.

### Option B: Convert every artifact into a symbolic expression

| Dimension | Assessment |
|---|---|
| Apparent API uniformity | High |
| Representation fidelity | Low |
| Scalability | Low |
| Soundness clarity | Low |

**Pros:** Reuses the existing symbolic problem and checkers directly.

**Cons:** Most neural and numerical artifacts cannot be converted exactly. Any
approximation would blur the difference between identities and sampled values.

### Option C: Typed artifacts with representation-specific backends

| Dimension | Assessment |
|---|---|
| Initial complexity | Medium |
| Representation fidelity | High |
| Scalability | High |
| Soundness clarity | High |

**Pros:** Keeps proof scope explicit, preserves native execution, and lets new
artifact types arrive independently.

**Cons:** Some problem and checker concepts remain backend-specific, and users
must choose a compatible pair.

## Trade-off analysis

Option C provides a common user-level concept without claiming that all
artifacts support the same mathematics. The stable layer is the artifact and
report vocabulary; evaluation and proof rules remain explicit per backend.
This is more honest than a single overly general problem class and more useful
than unrelated verification functions with incompatible outputs.

The initial dispatcher contains two explicit backend pairs. A verifier-backend
registry should replace that dispatch only when a third backend establishes the
actual common contract. Designing that registry now would freeze assumptions
before numerical and program artifacts exist.

## Consequences

- Analytical expressions, PyTorch callables, and future artifact types can share
  field naming and machine-readable outcomes.
- Current Python callers, JSON cases, benchmark digests, and symbolic checker
  semantics remain compatible.
- PyTorch users can express PDE, initial, and boundary residuals with automatic
  derivatives without making PyTorch a core dependency.
- A passing callable receives `INCONCLUSIVE`, not a numerical certificate.
- Callable residual operators are trusted Python code and are not accepted by
  the restricted JSON parser.
- Callable fields must be pointwise across the batch dimension. Cross-sample
  operations require a Jacobian-based checker with different cost and semantics.
- Gridded and generated-program artifacts still require separate security and
  numerical-semantics decisions.

## Action items

1. [x] Define `SolutionArtifact`, `SymbolicCandidate`, and `CallableCandidate`.
2. [x] Preserve the existing symbolic API through an internal artifact wrapper.
3. [x] Add PyTorch autodiff problems, surface conditions, and residual checking.
4. [x] Test passing, violated, boundary, non-finite, and mismatched cases.
5. [x] Publish a runnable mixed symbolic/callable example.
6. [ ] Add a balanced benchmark from trained PINNs and external solver outputs.
7. [ ] Design a non-executing serialization format for callable problem metadata.
8. [ ] Specify process isolation and resource limits before adding program artifacts.
9. [ ] Add interval and weak-form proof backends for explicitly supported classes.
