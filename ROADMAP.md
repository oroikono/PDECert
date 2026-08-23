# Roadmap

PDECert aims to become a practical verification layer between symbolic PDE
candidate generators and the people who need to trust their output. The project
is currently an early research prototype. This roadmap describes public project
outcomes, not a private development schedule.

## v0.1: dependable verification interface

The first release should let a user save one fully instantiated verification
case, run it locally, and understand why the result is proved, refuted, or
inconclusive.

Available:

- conservative `PROVED`, `REFUTED`, and `INCONCLUSIVE` outcomes;
- exact residual and initial/boundary-condition checks;
- off-grid numerical counterexamples and singularity witnesses;
- per-check symbolic deadlines with explicit incomplete reasons;
- explicit parameter roles, assumptions, and domain-aware sampling;
- a versioned JSON case format with restricted expression parsing;
- a command-line verifier with machine-readable reports.

Release criteria:

- operation and memory budgets for symbolic checks;
- multiple fields and coupled residual equations;
- stable error messages, documentation, and tests on supported Python versions.

## v0.2: natural-candidate benchmark

The second release will test whether the verifier is useful beyond constructed
examples.

- define a candidate-corpus format with problem, solver, model, prompt, and
  provenance fields;
- publish a pilot set of 20 natural outputs from symbolic solvers and open models;
- document a manual labeling and disagreement-resolution protocol;
- grow the first public benchmark to at least 100 candidates;
- compare fixed collocation, SymPy checks, and PDECert on false acceptance,
  false rejection, inconclusive rate, witness quality, and runtime;
- publish the benchmark and data card on Hugging Face.

## v0.3: integrations

- adapters for common SymPy and symbolic-regression workflows;
- a small agent tool that returns a certificate, counterexample, or inconclusive result;
- a reproducible notebook or Space for exploring benchmark cases;
- structured reports that candidate generators can use for repair.

## Longer-term research

- interval-based counterexample search for supported expression classes;
- explicit classical, weak, and other solution semantics;
- a posteriori error bounds for supported numerical settings;
- verifier-guided candidate repair using returned counterexamples;
- broader PDE families, systems, domains, and boundary operators.

## Where contributions help most

Real failure cases are more valuable than feature volume. Useful contributions
include a symbolic candidate that passes a common check but violates its stated
problem, a minimal new PDE example, an improvement to an inconclusive diagnostic,
or an adapter to a solver people already use.

The scope remains conservative: supported classes should be explicit, and
unsupported cases should stay inconclusive.
