# Changelog

All notable changes to PDECert are documented here.

## Unreleased

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
