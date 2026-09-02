# Limitations and threats to validity

PDECert is a conservative evaluation layer for machine-generated PDE solution
artifacts. It is not a general PDE solver, proof assistant, neural-network
certifier, or guarantee that a represented PDE problem is mathematically
well-posed. This document defines the claims that current releases support and
the claims that users must not infer from a report or benchmark result.

The governing rule is simple: evidence applies only to the declared obligations,
artifact, evaluator, and assumptions recorded in that run. It does not transfer
between artifacts or beyond that scope.

## Current supported scope

| Lane | Current interpretation | Strongest current positive result | Unsupported or excluded |
| --- | --- | --- | --- |
| Symbolic expressions | Classical strong-form residuals and represented initial or boundary conditions over declared real rectangular domains | `PROVED` when every represented obligation is discharged by exact symbolic evidence | Weak, entropy, or viscosity semantics; general irregular domains; unrepresented regularity, existence, uniqueness, or conservation obligations |
| Differentiable callables and PINNs | Pointwise PyTorch automatic-differentiation checks at deterministic finite samples | A replayable empirical counterexample can produce `REFUTED`; a passing run remains `INCONCLUSIVE` | Formal neural-network certification, global residual bounds, and solution-error guarantees |
| Matched cases | Separate reports for different artifacts bound to one trusted mathematical task | Per-lane evidence with no combined status | Transferring an exact symbolic result to a trained callable or treating unlike evaluators as equally strong |
| Agent proposals | Provenance and repair traces around a supported materialized artifact | The result of the underlying symbolic or callable verifier | Treating model self-critique, repair, or tool use as independent ground truth |
| Generated programs | A non-executing source artifact and deny-by-default sandbox contract | Safe rejection when no sandbox is configured | Executing untrusted source in the core package or claiming that declared sandbox capabilities are remote attestation |
| Rigorous bounds | A typed report schema for quantity, norm, scope, assumptions, and constants | Transport and validation of evidence emitted by an explicitly configured checker | A built-in validated numerical backend; version `0.1.1rc2` ships no native interval or a posteriori certification method |

The public problem-template path currently declares `classical_strong` solution
semantics. Other semantics appearing in the roadmap are research directions,
not implemented capabilities.

## What each decision means

### `PROVED`

`PROVED` means that every obligation represented in the evaluation context was
discharged by an accepted exact or rigorous-bound evidence event. In the
current built-in verifier, positive results come from exact symbolic checks.

It does not establish:

- existence or uniqueness of a solution to the original mathematical problem;
- regularity assumptions that were not encoded as obligations;
- equivalence between the encoded problem and a paper, prompt, or physical
  system from which it was transcribed;
- correctness of a different symbolic expression, callable model, checkpoint,
  discretization, or agent response; or
- a proof checked by a small formal proof-assistant kernel.

The trusted computing base includes PDECert, SymPy, Python, and the correctness
of the encoded operator and conditions. A bug in any of these can invalidate a
result.

### `REFUTED`

`REFUTED` means that at least one represented obligation has a concrete
symbolic or empirical witness, such as a domain singularity or a sampled point
whose residual exceeds the configured tolerance. The witness refutes that
candidate under the recorded evaluator configuration.

A floating-point witness can be affected by dtype, library versions, hardware,
and numerical conditioning. Important findings should be replayed at higher
precision or with an independent implementation when those effects could
change the conclusion.

### `INCONCLUSIVE`

`INCONCLUSIVE` is an intended result, not an error or evidence of correctness.
It covers symbolic expressions the current simplifier cannot decide, supported
checks that exceed configured limits or encounter an unsupported operation, and
callable artifacts whose finite samples did not reveal a violation. Inputs
outside a public representation contract, such as a weak-solution template, may
instead be rejected explicitly before a report is created.

No number of ordinary sampled passes is promoted to `PROVED`.

## Mathematical limitations

### Strong form only

Current obligations use pointwise classical derivatives. They do not define
weak forms, test-function spaces, trace theorems, entropy conditions, viscosity
solutions, or distributional derivatives. Discontinuous solutions, shocks,
and irregular boundary traces therefore fall outside the current proof scope,
even if a finite diagnostic can be computed.

### Residual evidence is not solution-error evidence

A small residual, or even a rigorous residual enclosure, is not automatically a
bound on the distance to the true solution. Such a conclusion requires a
problem-specific stability or a posteriori theorem, its constants, norms, and
assumptions. Consumers must inspect `bound.bound_type`; a
`UNIFORM_RESIDUAL` or `BOUNDARY_TRACE` bound is not a `SOLUTION_ERROR` bound.

### Domain and boundary representation

The built-in symbolic and callable problems use finite rectangular coordinate
domains. Callable conditions fix coordinate values to describe aligned initial
or boundary surfaces. General meshes, implicit geometries, curved or moving
boundaries, interfaces, periodic topology, and arbitrary Robin or variational
operators do not have a complete public representation and verifier path.

### Symbolic incompleteness

Zero-equivalence and singularity analysis are incomplete in general. Equivalent
expressions involving branch cuts, special functions, piecewise definitions,
parameter-dependent singularities, or difficult multivariate identities may
remain undecided. PDECert should return `INCONCLUSIVE` when its symbolic path
cannot justify a decision; this does not imply that SymPy can never return an
incorrect result.

### Parameter assumptions

Only the declared, supported real assumptions and parameter domains participate
in verification. Missing physical restrictions, incompatible units, correlated
parameters, and constraints not expressible by the current schema remain the
caller’s responsibility. Finite parameter samples do not prove a parametric
identity.

## Numerical and implementation limitations

- Callable evaluation assumes that each output row depends only on the
  corresponding input row. Training-mode batch operations, cross-sample
  attention, mutable state, randomness, and hidden preprocessing can invalidate
  pointwise automatic derivatives.
- Tolerances determine whether a sampled value becomes a counterexample. Values
  below tolerance are observed passes, not evidence that the exact residual is
  zero.
- The symbolic operation budget limits input-tree size, not intermediate
  expression growth or process memory. Symbolic timeouts depend on main-thread
  interval-timer support and are not a process-isolation boundary.
- Domain singularity enumeration is conservative but incomplete. Failure to
  enumerate a singularity is not a global regularity theorem.
- PyTorch results can vary with device, dtype, kernels, compiler settings, and
  version. The frozen Burgers and Fisher--KPP fixtures record their environments
  but do not prove platform-independent optimization or evaluation.
- Content digests establish byte identity, not scientific correctness,
  reproducibility of stochastic training, or equivalence of two artifacts.
- Atlas v2 transport validation checks that a frozen callable is consistent
  with its declared artifact, configuration, weight, and source-file digests.
  It does not prove that the referenced training run occurred. Full repository
  reproduction is a separate check, and mixed-record blind review and immutable
  release tooling are not yet implemented.
- A symbolic Atlas artifact records parsed field expressions beside a bound raw
  response. Structural validation does not establish that the parser or human
  transcription preserved the intended semantics of that response.

Restricted parsing reduces the attack surface for symbolic inputs, but complex
valid expressions may still consume substantial CPU or memory. Do not expose
the verifier to untrusted workloads without an outer process, resource, and
request-isolation policy.

## Benchmark threats to validity

### Coverage and external validity

The public pilot contains 20 candidate records and deliberately emphasizes
known symbolic failure modes. It is not representative of PDE solving, SciML,
LLM reasoning, or production workloads. Aggregate rates from this pilot are
descriptive and must not be generalized to a model family or user population.

The community Atlas is an intake corpus under active development. Record count
alone is not evidence of family, artifact, generator, or difficulty coverage.
Every public audit must publish its coverage matrix and unresolved blind spots.

### Construct and manufactured-case bias

Exact solutions and injected perturbations provide controlled tests, but can
make verification look easier than naturally generated failures. The trained
Burgers and Fisher--KPP fixtures are two small networks and two training setups.
The Fisher--KPP prompt exposes the expected traveling-front form through its
declared traces, and the preserved open-model proposal still awaits independent
human review. These fixtures test artifact and evidence contracts; they are not
claims of state-of-the-art PINN evaluation or representative LLM reasoning.

### Labels and reference uncertainty

Machine reports, model critiques, and user-approved amendments are not
independent labels. Benchmark claims require the published blind-review and
disagreement-adjudication protocol. Numerical reference fields also carry
discretization, solver, and convergence uncertainty and must not be presented
as exact truth without justification.

### Agent and hosted-model reproducibility

An agent trace establishes what was submitted to a verifier and how later
proposals were linked. It does not isolate the effect of the verifier, prove the
model’s reasoning, or measure broad agent accuracy. Hosted provider behavior is
not fully pinned by a Hub revision; deployment, routing, and server-side
configuration can change.

### Baseline fairness

Baselines must receive the same represented PDE, domain, conditions, parameters,
and artifact information. Comparing a PDE-only checker with a full
initial-boundary verifier measures obligation coverage as well as algorithmic
quality. Reports should separate those effects instead of collapsing them into
one accuracy number.

## Security and trust boundaries

- Candidate-free problem templates, problem definitions, evaluator settings,
  and labels are trusted inputs. A candidate must not be allowed to replace
  them.
- Agent raw outputs remain distinct from parsed artifacts and verification
  reports. Hashing or recording a response does not validate it.
- `ProgramCandidate` construction never executes source. The core package has
  no local `exec`, shell, or subprocess fallback for generated code.
- An external program sandbox must be audited independently. Its declared
  capabilities do not prove that syscall, network, filesystem, secret, timeout,
  and cleanup policies are actually enforced.
- PDECert reports are evidence records, not authorization decisions for safety-
  critical systems.

## Reproduction requirements

A result intended for review or publication should retain:

1. the exact problem template or case and its solution semantics;
2. the unedited candidate artifact and content digest;
3. evaluator name, version, configuration, evidence-report version, and package
   environment;
4. all assumptions, domains, tolerances, sampling settings, and random seeds;
5. the complete report, including abstentions, incomplete reasons, and
   witnesses—not only the top-level status;
6. training or generation provenance when applicable; and
7. independent labels and disagreement notes when a benchmark claim uses human
   ground truth.

Run manifests bind many of these files by digest, but reviewers must still
inspect whether the declared files faithfully represent the intended problem.

## Publication checklist

Before making a claim based on PDECert, state:

- which artifact and solution semantics were accepted;
- which obligations were represented and which were omitted;
- the evidence kind behind each decisive result;
- whether the result is a residual, boundary, or solution-error statement;
- all unsupported and inconclusive cases;
- corpus coverage, selection procedure, and label uncertainty; and
- the independent baseline or reproduction used to challenge the result.

See [`docs/evidence-reports.md`](docs/evidence-reports.md) for the machine
contract, [`ARCHITECTURE.md`](ARCHITECTURE.md) for extension boundaries, and
[`docs/research-landscape.md`](docs/research-landscape.md) for adjacent work and
novelty constraints. Update this document whenever a release changes an
evidence level, accepted solution semantics, security boundary, or benchmark
claim.
