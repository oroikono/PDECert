# Roadmap

PDECert aims to become a practical verification layer between PDE solution
generators and the people who need to trust their output. Those artifacts can
be symbolic expressions, differentiable models, numerical fields, or generated
solver programs. The project is currently an early research prototype. This
roadmap describes public project outcomes, not a private development schedule.

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
- named candidate fields and coupled residual systems;
- a versioned JSON case format with restricted expression parsing;
- a command-line verifier with machine-readable reports.

Release criteria:

- operation and memory budgets for symbolic checks;
- stable error messages, documentation, and tests on supported Python versions.

## v0.2: natural-candidate benchmark

The second release will test whether the verifier is useful beyond constructed
examples.

Available:

- a versioned candidate-corpus format with embedded cases, raw outputs, content
  digests, solver/model provenance, prompts or solver inputs, and annotation state;
- a 20-record pilot with 10 reproducible SymPy outputs and 10 pinned local
  open-model generations;
- a blind human-review protocol, resumable card runner, blank review form,
  guarded annotation importer, and clearly separated provisional comparison file;
- 20 independently reviewed pilot labels with retained comparison-stage
  amendments and public reviewer identifier;
- a label-gated report pipeline comparing fixed collocation, direct SymPy
  residual checks, and PDECert with error, abstention, witness, and runtime metrics;
- a committed pilot comparison bound to the exact labeled-corpus digest;
- a deterministic, digest-checked Hugging Face release builder with JSONL,
  dataset-card, benchmark-report, and manifest outputs;
- a public, immutable first benchmark release at
  [`oroikono/pdecert-pilot`](https://huggingface.co/datasets/oroikono/pdecert-pilot).

Next:

- grow the first public benchmark to at least 100 candidates;
- add harder cases where finite collocation and exact verification disagree.

## v0.3: integrations

- an explicit checker registry with immutable run configuration and validated
  obligation scope;
- a general solution-artifact protocol with symbolic and PyTorch-callable
  implementations;
- automatic-differentiation residual, initial-condition, and boundary-condition
  checks that conservatively abstain after sampled success;
- adapters for common SymPy and symbolic-regression workflows;
- a small agent tool that returns a certificate, counterexample, or inconclusive result;
- a reproducible notebook or Space for exploring benchmark cases;
- structured reports that candidate generators can use for repair.

## Longer-term research

- gridded numerical artifacts with refinement and convergence checks;
- generated solver programs behind a documented process-isolation boundary;
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
