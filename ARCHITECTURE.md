# Architecture

PDECert is a verification and evaluation package for candidate PDE solutions.
It is intentionally not a solver and not one universal theorem prover. The
package combines several evidence-producing backends while keeping their
assumptions and strength visible in one report vocabulary.

This document is the contributor map: it describes where new work belongs, the
boundaries it must preserve, and how a contribution becomes a reviewable
vertical slice. Mathematical decisions with lasting compatibility consequences
are recorded separately in [`docs/adr/`](docs/adr/).

## Design constraints

Five constraints govern the architecture:

1. **Evidence strength stays explicit.** Exact symbolic reasoning, rigorous
   numerical bounds, and empirical diagnostics are different products. A
   backend must not silently promote sampled success into proof.
2. **Artifacts keep their native semantics.** Symbolic expressions,
   differentiable callables, gridded fields, and generated programs require
   different evaluation rules and trust boundaries.
3. **Extensions are explicit.** Verification must not change because an
   unrelated package happened to be installed. Checkers and future adapters are
   supplied through ordered registries.
4. **Unsupported inputs abstain.** A narrow sound backend is more useful than a
   broad backend that overclaims. Unsupported cases return an inconclusive
   result with a reason.
5. **Benchmark evidence is reproducible.** Cases, model or solver provenance,
   review state, configuration, and output digests are part of the result.

## Package layers

```text
User interfaces
  CLI, Python API, reports
          |
Evaluation and release
  benchmark, corpus, labeling, release
          |
Verification orchestration
  checker registries, obligation scope, evidence aggregation
          |
Representation backends
  symbolic problems       callable/autodiff problems       future grid problems
          |                         |                              |
Candidate artifacts
  SymbolicCandidate        CallableCandidate               future GridCandidate
          |
Versioned specifications
  core data types, restricted JSON schemas, stable identifiers
```

Dependencies should point downward. Corpus and CLI code may use verification
backends; a verification backend must not depend on corpus annotation or command
line behavior. The diagram describes the intended responsibility flow; a few
legacy lazy imports in `core.py` preserve the existing public API and should not
be copied into new modules.

## Current module responsibilities

| Module | Responsibility | Extension rule |
| --- | --- | --- |
| `core.py` | Conservative statuses, symbolic problem types, witnesses, reports, and legacy verification entry point | Keep small. Changes affect every backend and require soundness-focused tests. |
| `artifacts.py` | Representation-neutral candidate identity plus concrete artifact types | Add an artifact only after its native evaluation semantics are understood. |
| `checks.py` | Symbolic checker protocol, registry, built-in checks, and evidence aggregation | New symbolic methods implement `Checker`; they do not add branches to the orchestrator. |
| `autodiff.py` | Callable problem description and PyTorch residual evaluation | Keep PyTorch optional and sampled success inconclusive. |
| `compiler.py` | Restricted lowering from retained operator sources to callable residuals | Reject unsupported semantics before evaluation; do not transfer evidence between artifacts. |
| `templates.py` | Candidate-free problem specifications and explicit symbolic bindings | Keep problem ownership separate from candidate provenance and preserve declared solution semantics. |
| `schema.py` | Restricted, versioned problem serialization | Never execute arbitrary code from a case file. Schema changes require migration and compatibility tests. |
| `corpus.py` | Versioned candidate and Atlas records with provenance | Raw model or solver outputs remain unedited. Annotation state stays distinct from machine evidence. |
| `labeling.py` | Blind review and guarded label import | Machine proposals are never represented as independent human ground truth. |
| `benchmark.py` | Evaluator comparisons and aggregate metrics | Report abstention and failure, not only accuracy. |
| `release.py` | Digest-bound public artifacts | Releases must rebuild deterministically from committed inputs. |
| `cli.py` | Stable command behavior and machine-readable output | CLI exit codes and report fields are public contracts. |

## Public and experimental surfaces

PDECert is alpha software. Stability is graduated rather than binary.

### Compatibility commitments

- versioned JSON inputs remain readable or receive an explicit migration path;
- report fields are not silently redefined;
- `PROVED`, `REFUTED`, and `INCONCLUSIVE` keep conservative meanings;
- empirical sampled success does not become `PROVED`;
- PyTorch remains optional for symbolic users;
- release manifests remain bound to content digests.

### Experimental extension surfaces

- checker and artifact protocols;
- callable problem metadata;
- future grid, interval, weak-form, and program backends;
- benchmark-method and external-generator adapters.

Experimental APIs may evolve, but changes still require migration notes and
focused tests. Once two independent external extensions use the same contract,
the project can consider stabilizing it.

## Contributor workstreams

Work is divided by scientific responsibility, not by file count.

### 1. Soundness core

Owns statuses, obligation identifiers, aggregation, evidence semantics, resource
limits, and report compatibility. These changes have the highest review burden.

Examples: enforcing evidence levels, validating witness scope, or adding a
process-level symbolic memory limit.

### 2. Artifact representations

Owns faithful representations of what a generator actually produced.

Examples: a gridded field with coordinates and discretization metadata, or a
non-executing description of a trained callable artifact. Representation work
must precede verification claims about that artifact.

### 3. Verification backends

Owns one explicitly supported mathematical decision procedure.

Examples: an Arb enclosure for a restricted expression grammar, a
`partial-CROWN` adapter, or an empirical weak-residual evaluator. Each backend
documents its assumptions, supported class, evidence strength, and abstention
boundary.

### 4. Benchmark science

Owns matched cases, natural candidate collection, failure taxonomy, independent
labels, baselines, metrics, and uncertainty reporting.

Examples: the same Burgers problem represented by an analytical expression and
an independently trained PINN, or a preregistered boundary-defect corruption.

### 5. Ecosystem integrations

Owns adapters that make existing SciML work evaluable without replacing its
native format.

Examples: PDEBench tensor loading, UFL weak-form translation, or a report adapter
for a scientific agent. Integrations stay optional and do not enlarge the core
dependency set without an ADR.

### 6. Developer experience

Owns packaging, documentation, examples, CI, reproducible environments, and
release automation. Developer-experience changes must still correspond to a
real user or contributor need.

## The vertical-slice rule

One contribution should deliver one observable capability end to end:

```text
documented mathematical scope
        +
implementation behind the correct extension boundary
        +
valid, invalid, and unsupported tests
        +
one minimal reproducible example or corpus fixture
        +
user-facing report or diagnostic
```

A large backend should be split into independently useful slices. For example,
a grid lane should land metadata validation before derivative reconstruction,
and derivative reconstruction before convergence or conservation metrics.

## Backend contract

Every proposed checker or evaluator answers these questions before code review:

1. What artifact and problem representation does it accept?
2. Which obligations can it address: PDE, initial, boundary, interface, or
   domain?
3. What mathematical semantics does it assume: classical, weak, entropy, or a
   narrower named class?
4. What evidence does it produce: exact symbolic evidence, rigorous bound, or
   empirical diagnostic?
5. What constitutes a replayable refutation witness?
6. Which inputs are unsupported, and how does the backend abstain?
7. Which dependency, precision, tolerance, partition, seed, or model version is
   required to reproduce the result?
8. What resource limit prevents one case from blocking an evaluation run?

## Benchmark contract

A public benchmark result must distinguish four layers:

- the unedited candidate artifact and its provenance;
- the mathematical problem and intended solution semantics;
- an independent reference label, including disagreement history;
- machine evidence emitted by each evaluator.

The benchmark should report unsafe proof rate, refutation recall, abstention,
witness replay rate, runtime, and certificate coverage where applicable. A
single aggregate accuracy number is insufficient.

## Daily development policy

Daily work advances one unmet release gate. It does not create activity-only
commits or split one coherent change into artificial pull requests. Before
starting, inspect open work to avoid duplication. At the end of a successful
slice, the repository should be easier for the next contributor to understand
or extend.

The preferred sequence is:

1. define or confirm the representation and contract;
2. add the smallest implementation with conservative behavior;
3. add adversarial and unsupported tests;
4. add one contributor-facing example;
5. publish one focused pull request after all relevant checks pass.

## Architectural decisions

- [`ADR-0001`](docs/adr/0001-plugin-first-extension-architecture.md): explicit
  checker registries and conservative aggregation.
- [`ADR-0002`](docs/adr/0002-general-solution-artifacts.md): typed candidate
  artifacts with representation-specific verification.
- [`ADR-0004`](docs/adr/0004-community-evolution-model.md): workstreams,
  vertical slices, and release gates for community evolution.
- [`ADR-0006`](docs/adr/0006-portable-operator-lowering.md): a declared subset
  of trusted operator sources can drive symbolic and callable lanes.
- [`ADR-0007`](docs/adr/0007-candidate-free-problem-templates.md): trusted
  problem templates are serialized independently from candidate artifacts.
