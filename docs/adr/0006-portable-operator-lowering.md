# ADR-0006: Portable operator lowering across symbolic and callable artifacts

**Status:** Proposed
**Date:** 2026-08-28
**Decider:** Orestis Oikonomou

## Context

PDECert stores field-referenced differential operators in its restricted
version 3 case format, but callable verification previously required a second,
handwritten `AutodiffProblem`. Duplicating a PDE in a symbolic expression and a
Python lambda creates an avoidable semantic-drift risk and makes it difficult
for generators, agents, and learned models to share one trusted task.

This is not an unclaimed problem space. UFL is a mature language for finite
element variational forms, and ModelingToolkit's `PDESystem` pursues a common
symbolic PDE interface across discretizers. TensorMesh exposes a PyTorch-native
mesh, assembly, sparse-solve, and differentiation workflow. CodePDE and
PDEAgent-Bench evaluate LLM-generated solver programs. PDECert's narrower need
is to preserve one trusted classical operator specification while evaluating
candidate artifacts through backends with different evidence strength.

## Decision

Add an explicit compiler from retained PDECert operator sources to the current
PyTorch `AutodiffProblem` representation.

The initial compiler:

- translates arithmetic, common scalar functions, `D`, and consistent `At`
  surfaces;
- differentiates evaluated expressions, not only bare field names;
- uses the case's coordinates, domains, field names, obligation names, and
  operator sources;
- ignores the case's symbolic candidate values when constructing the callable
  problem;
- rejects unsupported parameters, surface semantics, and functions before
  callable evaluation;
- leaves automatic-differentiation passes empirical and therefore
  `INCONCLUSIVE`.

The compiler is one-way. It does not claim that arbitrary Python residual
lambdas can be recovered as symbolic expressions, and it does not claim that
the current grammar is a universal PDE intermediate representation.

## Options considered

### Option A: Keep handwritten symbolic and callable problems

**Pros:** no compiler implementation and unlimited Python expressiveness.

**Cons:** duplicated mathematics can drift; agent and benchmark tasks cannot
reuse one checked operator source.

### Option B: Replace the schema immediately with UFL or another external DSL

**Pros:** mature operator and finite-element semantics.

**Cons:** large compatibility and dependency change; UFL's variational-form
scope does not by itself define PDECert's evidence, artifact, or agent contract.

### Option C: Lower a declared subset of the existing restricted operators

**Pros:** demonstrates a real shared specification without changing saved
cases, adding a core dependency, or pretending unsupported semantics are
portable.

**Cons:** the subset is initially classical, rectangular, and parameter-free.

## Consequences

- A symbolic generator, trained callable, and symbolic agent tool can target
  the same trusted case definition.
- Constraint names and boundary surfaces stay aligned across exact and
  empirical lanes.
- PyTorch remains optional and no generated code is executed.
- Unsupported functions and ambiguous surfaces fail during compilation instead
  of producing misleading residuals.
- The callable backend still provides diagnostics, not formal proof.
- General parameters, weak forms, irregular domains, vector/tensor fields, and
  external DSL adapters remain separate design decisions.

## Action items

1. [x] Lower a documented expression subset into `AutodiffProblem`.
2. [x] Support derivatives of composite pointwise expressions.
3. [x] Test valid, refuted, and unsupported cases.
4. [x] Publish one single-source symbolic/callable example.
5. [ ] Add a stable, candidate-free problem-template serialization after two
   independent integrations exercise the contract.
6. [ ] Evaluate a UFL import adapter for a narrow classical/variational subset.
7. [ ] Define batched parameter semantics before lowering parameterized cases.

## References

- [UFL documentation](https://docs.fenicsproject.org/ufl/main/)
- [ModelingToolkit `PDESystem`](https://docs.sciml.ai/ModelingToolkit/v8.75/systems/PDESystem/)
- [TensorGalerkin](https://arxiv.org/abs/2602.05052)
- [CodePDE](https://arxiv.org/abs/2505.08783)
- [PDEAgent-Bench](https://arxiv.org/abs/2605.09636)
