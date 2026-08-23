# Changelog

All notable changes to PDECert are documented here.

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

The pilot is designed to exercise known failure modes. Its measurements are
descriptive and are not population-level performance estimates.
