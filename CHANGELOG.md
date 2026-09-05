# Changelog

All notable changes to PDECert are documented here.

## Unreleased

### Added

- An explicit Atlas baseline-adapter contract and deterministic full-condition
  fixed-collocation runner for symbolic records, with empirical-only pass/fail
  semantics, replayable numerical-threshold witnesses, structured callable
  abstention, resource bounds, a CLI, and a versioned report schema.

- Strict loading and digest-bound descriptive summaries for typed Atlas
  evaluations, with per-problem artifact views and no aggregate verdict or
  truth-label inference.

- A versioned Atlas v2 evaluator and CLI for symbolic and restricted frozen-
  callable records, with exact content binding, explicit reproduction options,
  complete per-record reports, and no cross-artifact status aggregation.
- A digest-bound Atlas v2 blind-review contract with artifact-aware review
  bases enforced by the importer, loader, writer, and public schema, plus atomic
  annotation import that preserves candidate and problem bytes.
- A typed Atlas v2 record contract for candidate-free symbolic and frozen-
  callable artifacts, with strict file digests, provenance, representation
  compatibility checks, public schemas, and a mixed Fisher--KPP corpus that
  validates without importing PyTorch.
- A portable versioned JSON contract for frozen CPU float64 dense-tanh
  callables, including strict structural validation, immutable loading,
  resource-bounded decoding, evaluation-only PyTorch materialization, portable
  integrity records, schemas, tests, and contributor documentation.
- A Fisher--KPP matched evaluation that preserves a Qwen3 symbolic proposal,
  trains a separate PINN from PDE and trace targets, digest-binds the resulting
  artifact and sources, and records the held-out empirical counterexample
  without transferring exact symbolic evidence to the callable lane.

## 0.1.1rc2 - 2026-09-01

### Changed

- Runtime version reporting now comes from installed package metadata, keeping
  `pdecert.__version__` aligned with wheel and source-distribution versions.
- Exact third-party checkers retain compatibility through synthesized evidence
  events, while rigorous-bound claims now require machine-readable quantity,
  scope, assumptions, and constants.

### Added

- An explicit limitations and threats-to-validity contract covering decision
  scope, unsupported mathematics, numerical and security boundaries, benchmark
  validity, and publication requirements.
- A deterministic `pdecert quickstart` command that demonstrates exact proof,
  empirical refutation, conservative abstention, and a recorded agent repair
  trace from clean wheel and source-distribution installs.
- A versioned decision-report contract with obligation-level exact, rigorous,
  empirical, and abstention events, strict JSON loading, and a canonical schema.
- A restricted compiler from retained version 3 operator sources to PyTorch
  automatic-differentiation problems, with explicit parameter and boundary
  limitations, composite-expression derivatives, tests, and a single-source
  heat example.
- A separately versioned, candidate-free problem-template format with strict
  symbolic binding, direct callable lowering, CLI validation, a JSON schema,
  architecture decision, and a reproducible heat example.
- A backend-neutral, digest-bound run-manifest format with path-safe bundle
  validation, immutable evaluator configuration, CLI support, a JSON schema,
  tamper tests, and a complete symbolic example bundle.

## 0.1.1rc1 - 2026-08-27

### Changed

- Restricted source distributions to an explicit public-file allowlist so local
  environments and unrelated checkout files cannot leak into release archives.
- Added clean wheel and source-install smoke tests and updated GitHub Actions to
  its current Node.js runtime.

### Added

- A nonlinear viscous-Burgers matched fixture pairing an exact symbolic
  traveling wave with an independently trained, frozen PINN, including a
  restricted JSON loader, training provenance, tests, and a reproducible report.
- A bounded real-model agent smoke runner that records provider, model revision,
  prompt, decoding request, environment, raw verifier interactions, and the
  reproducibility limits of hosted inference.
- Recorded verifier-guided agent sessions, cross-model behavioral metrics, and
  an optional real smolagents `ToolCallingAgent` integration.
- A non-executing `ProgramCandidate`, deny-by-default sandbox protocol, bounded
  JSON output contract, and generated-program isolation decision record.
- Machine-readable decision-evidence levels distinguishing exact identities,
  rigorous bounds, and empirical evaluations, with checker-contract enforcement
  that prevents sampled evidence from proving obligations.
- An explicit, versioned Atlas coverage taxonomy for PDE families, solution
  artifact types, and spatial dimensions, with CLI summaries and guarded
  preservation through review import.
- A configurable structural operation budget for symbolic domain and identity
  checks, with a finite CLI default and explicit inconclusive reasons.
- A contributor-facing PDE Failure Atlas protocol, structured issue intake,
  modular record bundles, and `pdecert corpus validate` coverage summary.
- A general `SolutionArtifact` protocol with concrete symbolic and differentiable
  callable candidates.
- Optional PyTorch automatic-differentiation residual checks for PDE, initial,
  and boundary obligations.
- A typed `verify_artifact` entry point that rejects incompatible problem and
  artifact representations.
- A runnable heat-equation example covering exact and perturbed callable fields.
- An ordered, immutable checker registry with a public context, result, and
  checker protocol.
- Built-in domain, exact-identity, and off-grid checks implemented through the
  same extension contract available to external checkers.
- An architecture decision record and runnable polynomial-checker example.

## 0.1.0 - 2026-08-24

First research release of the verifier and the version 1 pilot benchmark.

### Verifier

- Conservative `PROVED`, `REFUTED`, and `INCONCLUSIVE` outcomes.
- Exact residual and initial/boundary-condition checks.
- Off-grid numerical counterexamples and singularity witnesses.
- Explicit parameter assumptions, named fields, and coupled systems.
- A versioned JSON case format and machine-readable command-line reports.

### Pilot benchmark

- 20 natural symbolic PDE candidates: 10 reproducible SymPy outputs and 10
  pinned local open-model generations.
- 20 independently reviewed labels, with 10 valid and 10 invalid candidates.
- Reproducible comparisons against fixed collocation and direct SymPy residual
  simplification.
- A deterministic Hugging Face bundle containing the labeled JSONL, dataset
  card, benchmark report, and checksum manifest.
- Public release at
  [`oroikono/pdecert-pilot`](https://huggingface.co/datasets/oroikono/pdecert-pilot),
  with immutable revision
  [`db690f9b`](https://huggingface.co/datasets/oroikono/pdecert-pilot/commit/db690f9b161762ea288dd5dfb4b6b2f999c48e03).

The pilot is designed to exercise known failure modes. Its measurements are
descriptive and are not population-level performance estimates.
