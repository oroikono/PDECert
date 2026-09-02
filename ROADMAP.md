# Roadmap

PDECert aims to become a practical verification layer between PDE solution
generators and the people who need to trust their output. Those artifacts can
be symbolic expressions, differentiable models, numerical fields, or generated
solver programs. The project is currently an early research prototype. This
roadmap describes public project outcomes, not a private development schedule.

Work is organized by the package layers and contributor workstreams in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Progress is credited when an outcome has
implementation, tests, documentation, and reproducible evidence—not for commit,
pull-request, or corpus-record counts alone.

## First publishable cross-artifact release

The next research release should test one central hypothesis: whether explicit
evidence strength and abstention reveal failures that siloed symbolic and SciML
evaluators miss. The required gates are ordered so contributors can land small,
independently useful slices.

1. **Evidence semantics:** reports distinguish exact symbolic evidence,
   rigorous bounds, empirical diagnostics, and missing evidence without
   changing conservative statuses.
2. **Community architecture:** package boundaries, backend contracts,
   compatibility expectations, and contribution definitions of done are public.
3. **Matched-case contract:** one mathematical case can bind multiple candidate
   artifacts without pretending that their evaluators have equal guarantees.
4. **Natural symbolic/callable pairs:** the corpus includes independently
   generated symbolic outputs and trained callable models on matched problems.
5. **Gridded artifact lane:** numerical and neural-operator fields have explicit
   coordinates, discretization metadata, and non-autodiff evaluation semantics.
6. **Realistic coverage:** nonlinear, coupled, multidimensional, discontinuous,
   and irregular-domain limitations are represented and reported.
7. **Failure taxonomy:** corruptions and natural failures cover boundary,
   initial-condition, parameter, phase, singularity, localization, and
   representation errors.
8. **Baseline suite:** SymPy, fixed collocation, reference-field metrics, and
   applicable established SciML evaluators run through reproducible adapters.
9. **Rigorous backend:** at least one external or native validated backend is
   independently reproduced with explicit assumptions and certificate scope.
10. **Public audit:** independent labels, disagreement adjudication, uncertainty
    reporting, immutable artifacts, and a complete baseline report are released.

These gates replace a raw 100-record target as the primary research milestone.
Corpus growth still matters, but only together with artifact, family, origin,
and evaluator-disagreement coverage.

Two nonlinear symbolic/trained-callable fixtures now exercise this gate: a
viscous Burgers traveling wave and a Fisher--KPP traveling front. The latter
binds a preserved open-model response to a separately trained PINN through the
portable frozen-callable contract and exposes a held-out residual failure even
after low training loss. Atlas v2 now binds the Fisher--KPP symbolic and frozen
callable artifacts as first-class, digest-checked records under one candidate-
free problem ID. The open-model prompt substantially cues the expected
traveling front, both mixed records remain pending, and the mixed-record review,
baseline, and immutable release paths are not complete. The gate therefore
remains incomplete pending independent review and broader documented coverage
across more than one natural matched problem.

## v0.1: dependable verification interface

The first release should let a user save one fully instantiated verification
case, run it locally, and understand why the result is proved, refuted, or
inconclusive.

Available:

- conservative `PROVED`, `REFUTED`, and `INCONCLUSIVE` outcomes;
- exact residual and initial/boundary-condition checks;
- off-grid numerical counterexamples and singularity witnesses;
- per-check symbolic deadlines with explicit incomplete reasons;
- configurable input-expression operation budgets that abstain before expensive
  symbolic work without disabling numerical counterexamples;
- explicit parameter roles, assumptions, and domain-aware sampling;
- named candidate fields and coupled residual systems;
- a versioned JSON case format with restricted expression parsing;
- a command-line verifier with machine-readable reports.

Release criteria:

- enforceable memory budgets for intermediate symbolic work;
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
- a restricted compiler that reuses one trusted operator source for symbolic
  and PyTorch-callable evaluation without combining their evidence;
- a versioned, candidate-free problem template with explicit symbolic binding,
  callable lowering, CLI validation, and unchanged case-schema compatibility;
- a digest-bound run manifest for problem, candidate, evaluator configuration,
  environment, and report identity across symbolic, callable, and agent lanes;
- automatic-differentiation residual, initial-condition, and boundary-condition
  checks that conservatively abstain after sampled success;
- a portable, non-executing frozen dense-MLP artifact with strict shape,
  finite-value, and content-integrity validation;
- adapters for common SymPy and symbolic-regression workflows;
- a small agent tool that returns a certificate, counterexample, or inconclusive result;
- a provider-backed smolagents runner with complete proposal, rejection, and
  repair traces, plus one public end-to-end smoke result;
- a reproducible notebook or Space for exploring benchmark cases;
- structured reports that candidate generators can use for repair.

## Longer-term research

- gridded numerical artifacts with refinement and convergence checks;
- one audited production sandbox adapter for the deny-by-default generated-program contract;
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
