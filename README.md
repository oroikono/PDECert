<p align="center">
  <img src="assets/pdecert-icon.png" alt="PDECert project icon" width="200">
</p>

<h1 align="center">PDECert</h1>

<p align="center"><strong>Proof-carrying PDE solutions.</strong></p>

Machine-generated PDE solutions should come with a certificate or a concrete
counterexample.

PDECert is an extensible verification framework for symbolic expressions and
differentiable callable fields. It checks the PDE residual together with initial
and boundary conditions and returns one of three outcomes:

- `PROVED` when every current obligation has exact or rigorous-bound evidence;
- `REFUTED` when it finds a singularity or a numerical counterexample;
- `INCONCLUSIVE` when the available checks cannot decide.

Numerical sampling is only used to refute. Passing sampled points is never
treated as a proof.

Every decisive report also states its `decision_evidence`: `EXACT`,
`RIGOROUS_BOUND`, or `EMPIRICAL`. Exact identities and future validated bounds
may prove represented obligations. Floating-point and autodiff checks are
empirical; they can expose a violation but cannot prove a passing candidate.
Versioned reports also retain one `evidence_events` record per checked
obligation so consumers can distinguish exact discharge, structured bounds,
empirical counterexamples, sampled passes, and abstentions. See the
[`report contract`](docs/evidence-reports.md).

> [!IMPORTANT]
> This is a research prototype, not a general theorem prover. The current
> `PROVED` result applies only to symbolic obligations and domain checks
> represented by the input problem. Passing callable samples is always
> `INCONCLUSIVE`; sampling can refute but cannot prove.

> [!NOTE]
> Failure of symbolic simplification is not a rejection. PDECert records the
> undecided obligation and returns `INCONCLUSIVE` unless another checker finds a
> witness. See [ADR-0003](docs/adr/0003-decision-evidence-levels.md).

The complete
[limitations and threats-to-validity statement](LIMITATIONS_AND_THREATS_TO_VALIDITY.md)
defines what each status establishes, unsupported mathematical semantics,
security boundaries, and the claims that must not be inferred from the current
pilot or trained fixture.

## Five-minute quickstart

Install the release candidate and run the complete offline demonstration:

```bash
python -m pip install pdecert==0.1.1rc2
pdecert quickstart
pdecert quickstart --json > pdecert-quickstart.json
```

It reproduces an exact symbolic proof, a boundary counterexample, a sampled
pass that correctly remains `INCONCLUSIVE`, and a recorded agent proposal-to-
repair trace. No repository checkout, optional dependency, credential, network
call, or model API is used. See the [walkthrough and scope](docs/quickstart.md).

## Why this exists

PDE solvers and language models can produce expressions that look convincing
and have a small residual on a fixed grid. That does not guarantee that they
satisfy the full problem. A candidate can fail at a boundary, hide a pole
between collocation points, or work only at one parameter value.

The project does not claim that PDE residual checking, PINN certification, or
structured PDE specifications are new. The
[research landscape](docs/research-landscape.md) records the closest prior work,
the claims it rules out, and the narrower cross-artifact benchmark hypothesis
that PDECert is testing.

## PDE Failure Atlas

The [PDE Failure Atlas](corpus/ATLAS.md) turns those failures into a public,
versioned corpus. Each record binds an unedited candidate to its stated problem,
origin metadata, human review, and machine evidence. Contributors can propose a
symbolic, PINN, neural-operator, numerical, or generated-program case without
first learning the corpus schema by using the
[failure-case issue form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml).

Versioned corpus files can be checked locally before review:

```bash
pdecert corpus validate corpus/pilot.json
pdecert corpus validate corpus/community
```

Open-model batches follow a
[predeclared, resumable collection protocol](docs/atlas-open-model-collection.md)
that retains raw responses and accounts for outputs that cannot be materialized.

The current public pilot has 20 records. The next coverage milestone is a
100-record community release spanning multiple PDE families, artifact types,
generators, and checker disagreements. Record count is not treated as evidence
of representativeness; every release reports its coverage and known blind spots.

The included experiment contains seven small heat-equation cases. Among the
five deliberately wrong candidates, the initial results are:

| Check | Wrong candidates accepted as valid |
| --- | ---: |
| PDE-only fixed collocation | 5 / 5 |
| Fixed collocation with initial/boundary conditions | 4 / 5 |
| SymPy PDE-only candidate check | 1 / 5 |
| PDECert | 0 / 5 |

PDECert refutes four and leaves the below-tolerance case `INCONCLUSIVE`. This is
a small adversarial experiment, not yet evidence of broad benchmark performance.

## Install and run

PDECert supports CPython 3.10 through 3.14. Every supported version runs the
same lint and unit-test gates in continuous integration.

```bash
git clone https://github.com/oroikono/PDECert.git
cd PDECert
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m experiments.adversarial_heat
python -m experiments.coupled_wave
python -m experiments.sigs_poisson_gauss
```

Install the optional PyTorch backend to check callable models and PINNs:

```bash
pip install -e ".[dev,autodiff]"
python -m examples.autodiff_heat
python -m experiments.mixed_artifact_smoke
python -m experiments.trained_burgers_pair
```

The trained Burgers example pairs an exact symbolic traveling wave with a
separately trained, frozen PINN stored as architecture-restricted JSON. Exact
evidence remains confined to the symbolic lane; sampled callable checks can
refute or abstain, never prove. See
[`docs/trained-callable-pairs.md`](docs/trained-callable-pairs.md).

The installed command accepts one versioned JSON case and prints a stable JSON
report:

```bash
pdecert verify examples/exact_heat.json
pdecert verify examples/exact_heat.json --output report.json
pdecert verify examples/exact_heat.json --symbolic-timeout 5
pdecert verify examples/exact_heat.json --max-expression-ops 10000
```

Exit code `0` means `PROVED`, `1` means `REFUTED`, `2` means `INCONCLUSIVE`,
and `64` reports an unreadable or invalid input file. A non-zero result is not
automatically a software failure; consumers should also read `report.status`.
The CLI gives each singularity and symbolic-identity check two seconds by
default. It also admits at most 10,000 structural operations to each symbolic
domain or identity check. An expression above that limit skips the expensive
symbolic operation, records the reason under `report.incomplete_reasons`, and
cannot produce `PROVED`. Off-grid evaluation still runs and can return a concrete
counterexample. The operation count limits input complexity; it does not bound
intermediate expression growth or process memory. A deadline, unsupported
deadline environment, or undecidable symbolic operation is likewise recorded
as incomplete.

## Small example

```python
import sympy as sp

from pdecert import Constraint, Problem, verify

x, t = sp.symbols("x t", real=True)
candidate = sp.exp(-sp.pi**2 * t) * sp.sin(sp.pi * x)

problem = Problem(
    name="heat equation",
    variables=(x, t),
    domains={x: (0.0, 1.0), t: (0.0, 1.0)},
    pde_residuals=(
        Constraint("PDE", sp.diff(candidate, t) - sp.diff(candidate, x, 2)),
    ),
    conditions=(
        Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
        Constraint("left boundary", candidate.subs(x, 0)),
        Constraint("right boundary", candidate.subs(x, 1)),
    ),
)

report = verify(problem, (candidate,), symbolic_timeout=2.0)
print(report.status)  # Status.PROVED
```

## PyTorch and PINN-style fields

`CallableCandidate` accepts named PyTorch functions or modules whose input has
shape `(points, variables)`. An `AutodiffProblem` defines residual operators in
terms of fields, coordinates, and automatic derivatives. Conditions can fix a
coordinate to describe initial or boundary surfaces. A callable must evaluate
each row independently; cross-sample attention and training-mode batch
operations do not represent pointwise PDE derivatives under this checker.

```python
import torch

from pdecert import (
    AutodiffConstraint,
    AutodiffProblem,
    CallableCandidate,
    verify_artifact,
)


def field(points):
    x, t = points[:, 0:1], points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


problem = AutodiffProblem(
    name="heat equation",
    variables=("x", "t"),
    domains={"x": (0.0, 1.0), "t": (0.0, 1.0)},
    pde_residuals=(
        AutodiffConstraint(
            "heat PDE",
            lambda value: value.derivative("u", "t")
            - value.derivative("u", "x", order=2),
        ),
    ),
)
artifact = CallableCandidate.from_mapping({"u": field}, dtype="float64")
report = verify_artifact(problem, artifact, tolerance=1e-9)
print(report.status)  # INCONCLUSIVE: sampled success is not a proof
```

The complete example includes the initial condition, both boundary conditions,
and a perturbed field that is refuted with a concrete point. See
[`examples/autodiff_heat.py`](examples/autodiff_heat.py) and
[`ADR-0002`](docs/adr/0002-general-solution-artifacts.md).

### One operator source for symbolic and callable candidates

A candidate-free template keeps the trusted task separate from any model or
solver output:

```python
from pdecert import bind_symbolic_candidate, load_template, verify

template = load_template("examples/heat-template.json")
case = bind_symbolic_candidate(
    template,
    {"u": "exp(-pi**2*t)*sin(pi*x)"},
)
report = verify(case.problem, case.candidate_fields)
```

The same template can be compiled for a separately supplied PyTorch callable.
It defines classical strong, pointwise obligations; it does not contain a
candidate and cannot transfer exact evidence from one artifact to another.
Validate a public task with `pdecert template validate TEMPLATE.json`. See the
[`problem-template contract`](docs/problem-templates.md),
[`heat template`](examples/heat-template.json), and
[`two-lane example`](examples/problem_template.py).

Version 3 constraint expressions can also be compiled into an
`AutodiffProblem`. This removes the handwritten duplicate between a trusted
symbolic PDE specification and its callable residual operators:

```python
from pdecert import compile_autodiff_problem, load_case

case = load_case("examples/exact_heat.json")
callable_problem = compile_autodiff_problem(case)
```

The compiler translates the trusted operators, not the saved symbolic
candidate. A separately trained PINN or other pointwise PyTorch field is still
required. Its sampled success remains `INCONCLUSIVE`; exact evidence from the
symbolic lane is never transferred to the callable artifact. The initial
lowering is parameter-free and classical, and rejects ambiguous boundary
surfaces or unsupported functions before model evaluation. See the
[`portable operator lowering`](docs/operator-compiler.md) guide and
[`examples/compiled_heat.py`](examples/compiled_heat.py).

To evaluate multiple representations of the same mathematical problem without
combining their evidence, use a [`MatchedCase`](docs/matched-cases.md). The
[`matched heat example`](examples/matched_heat.py) keeps exact symbolic and
sampled callable results in separate lanes and intentionally reports no overall
status.

## JSON cases

Version 3 of the case format stores a fully instantiated problem: declared real
variables, rectangular domains, named candidate fields, residual expressions,
and conditions. It also distinguishes parameter symbols from coordinates and
records sign, nonzero, or integer assumptions. The canonical shape is defined in
[`schema/problem-v3.schema.json`](schema/problem-v3.schema.json), with complete
single-field and coupled examples in [`examples/`](examples/).

Version 1 and version 2 inputs remain readable. Their positional candidate
expressions are assigned stable names such as `candidate_0` when loaded. New
files are written as version 3.

Candidate-free tasks use the separate version-1 template format rather than a
partially populated case. This leaves existing case semantics unchanged.

## Reproducible run bundles

A versioned run manifest binds one problem template, candidate artifact,
evaluator configuration, environment, and report by SHA-256. Validate the
complete example from any checkout:

```bash
pdecert run validate examples/heat-run-manifest.json
```

The command detects changed or missing files, path traversal, invalid template
semantics, candidate-field mismatch, and non-standard JSON reports. Its scope
is deliberately `content_identity_only`: matching hashes do not establish
authorship, trusted execution, or mathematical correctness. See the
[`run-manifest guide`](docs/run-manifests.md) and
[`ADR-0008`](docs/adr/0008-digest-bound-run-manifests.md).

Named fields make coupled systems explicit:

```json
"fields": {
  "u": "sin(pi*x)*cos(pi*t)",
  "v": "cos(pi*x)*sin(pi*t)"
}
```

Each coupled residual and condition is materialized as an expression. Schema v3
binds those expressions to the named candidates with two restricted
operators: `D(u, x, 2)` differentiates field `u` twice with respect to `x`, and
`At(u, x, 0)` evaluates it at the boundary `x = 0`. PDECert recomputes every
obligation from the field expressions and performs domain analysis on every
field. Derivative orders are limited to eight during input parsing. A field name
is included in any singularity witness.

```json
"pde_residuals": [
  {"name": "u_t - v_x", "expression": "D(u, t) - D(v, x)"},
  {"name": "v_t - u_x", "expression": "D(v, t) - D(u, x)"}
]
```

Parameters are declared by name with a list of assumptions:

```json
"parameters": {
  "k": ["positive", "nonzero"],
  "n": ["integer", "positive"]
}
```

Every parameter must also appear in `variables` and have a compatible finite
interval in `domains`. Variables omitted from `parameters` are treated as
coordinates. Integer parameters are sampled only at integer values.

```python
from pdecert import load_case, verify

case = load_case("examples/coupled_wave.json")
report = verify(case.problem, case.candidate_fields)
print(report.to_dict())
```

### Extending verification

Verification stages implement a public `Checker` protocol and run through an
ordered, immutable `CheckerRegistry`. Built-in and external checkers receive the
same `CheckContext` and return partial evidence as a `CheckResult`. The
orchestrator accepts proof evidence only for obligations already defined by the
problem and requires a concrete witness for refutation. New checkers should
attach an `EvidenceEvent` for each claimed proof, counterexample, sampled pass,
or abstention; rigorous claims additionally require structured `BoundEvidence`.

Registries are supplied explicitly so installing an unrelated package cannot
silently change verification behavior:

```python
from examples.polynomial_checker import ExpandedPolynomialChecker
from pdecert import default_checker_registry, verify

registry = default_checker_registry().with_checker(
    ExpandedPolynomialChecker(),
    before="exact_identity",
)
report = verify(problem, candidate_fields, checker_registry=registry)
```

The extension API is experimental in version 0.1. See
[`ADR-0001`](docs/adr/0001-plugin-first-extension-architecture.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) before implementing a checker.
The complete polynomial example uses only public PDECert objects and can serve
as the starting point for an external package.

Expression strings use a deliberately restricted arithmetic grammar. Declared
variables and fields, numeric literals, `pi`, `E`, `D`, `At`, and a documented
set of SymPy functions are accepted. Attribute access, imports, indexing,
unknown names, and keyword arguments are rejected before parsing. Version 1 and
2 residuals remain fully instantiated for backward compatibility; version 3
can bind operators and conditions directly to named fields.

## Candidate corpus

The natural-candidate benchmark uses a separate versioned format defined in
[`schema/corpus-v1.schema.json`](schema/corpus-v1.schema.json). Each record
contains:

- a complete schema-v3 verification case;
- the unedited raw solver or open-model output and its SHA-256 digest;
- producer, version, model or solver identifier, revision, source URL, license,
  generation timestamp, and the exact prompt or solver input;
- a structured annotation state with verdict, failure modes, rationale, and
  annotator identifiers.

The digest detects accidental or later changes to the stored output; it does not
by itself prove who generated the output. That claim remains grounded in the
recorded provenance and reproducible collection procedure.

[`corpus/pilot.json`](corpus/pilot.json) is the canonical pilot document. It is
paired with the exact collection method and summary in
[`corpus/README.md`](corpus/README.md). The current pilot contains 20 real runs:
10 SymPy solver outputs and 10 local generations from a pinned 4-bit Qwen3-0.6B
model. The completed blind review labels 10 records valid and 10 invalid under
the public reviewer identifier `oroikono`; the post-review amendments and exact
counterexample are retained in
[`corpus/review-comparison.json`](corpus/review-comparison.json).

```python
from pdecert import load_corpus

corpus = load_corpus("corpus/pilot.json")
print(len(corpus["records"]))
```

To reproduce collection on Apple silicon:

```bash
pip install -e ".[dev,collection]"
python -m experiments.collect_pilot
```

Human labels follow the blind-review and disagreement procedure in
[`corpus/LABELING.md`](corpus/LABELING.md). The importer requires an explicit
confirmation of independent review and never overwrites the source corpus. A
blank form and clearly separated provisional comparison file are provided; the
provisional file is not ground truth.

For a resumable blind pass with one card at a time and a progress bar:

```bash
python -m experiments.review_corpus
```

The same runner accepts a modular Atlas directory. Keep its resumable review
outside the tracked corpus:

```bash
python -m experiments.review_corpus corpus/community \
  --output private-reviews/community-review.json
```

Once every annotation is complete, run the reproducible comparison report:

```bash
python -m experiments.run_benchmark corpus/pilot.json \
  --output results/pilot-benchmark.json
```

The report compares fixed full-condition collocation, direct SymPy residual
simplification, and PDECert. It records false acceptance, false rejection,
inconclusive rate, witness coverage, per-record outcomes, and runtime. The
report binds those results to a corpus digest and records the runtime
environment. The command refuses to run while any annotation is pending.

The committed pilot report is
[`results/pilot-benchmark.json`](results/pilot-benchmark.json):

| Method | Accuracy | False accept | False reject | Inconclusive | Invalid witness |
|---|---:|---:|---:|---:|---:|
| Fixed collocation | 100% | 0% | 0% | 0% | 0% |
| Direct SymPy residual | 65% | 0% | 0% | 35% | 0% |
| PDECert | 100% | 0% | 0% | 0% | 100% |

These are descriptive results on 20 designed pilot records, not estimates of
performance on a broader solver or model population. Accuracy counts an
inconclusive outcome as incorrect; witness rate uses the 10 invalid records as
its denominator.

The labeled data and report are public in the
[`oroikono/pdecert-pilot`](https://huggingface.co/datasets/oroikono/pdecert-pilot)
dataset. The first release is fixed at Hub revision
[`db690f9b`](https://huggingface.co/datasets/oroikono/pdecert-pilot/commit/db690f9b161762ea288dd5dfb4b6b2f999c48e03)
and corpus digest
`4be9178edd30fcc561f21e83375713f4b38338484d75a0c8a7c8088e9c4369fb`.

Build the Hugging Face-ready release only after that report exists:

```bash
python -m experiments.build_release corpus/pilot.json \
  --benchmark results/pilot-benchmark.json \
  --output dist/pdecert-pilot
```

The release builder independently checks the corpus digest, row IDs, truth
labels, and aggregate metrics. It emits deterministic JSONL, a dataset card,
the full report, and a checksum manifest, and refuses pending labels or a
nonempty destination. Follow [RELEASE.md](RELEASE.md) for the final review,
Hub upload, viewer check, and immutable-link checklist.

## LLM and agent workflows

Agents can submit candidate artifacts, receive a counterexample or conservative
verification report, and propose a linked repair. The
[agent integration](docs/agent-integration.md) preserves raw-output provenance,
keeps the trusted problem outside agent-controlled symbolic tool input, and
includes an optional real smolagents `ToolCallingAgent` runner. Cross-model
summaries measure verifier calls and repair-to-`PROVED` behavior; they do not
relabel those outcomes as human ground-truth accuracy. Generated program
execution remains disabled by default. See
[`examples/agent_repair_loop.py`](examples/agent_repair_loop.py) for a
framework-free proposal and repair trace.

## Generated solver programs

`ProgramCandidate` records untrusted Python source and declared output fields,
but constructing it never executes the source. `execute_program_candidate`
fails closed unless an explicitly configured `ProgramSandbox` declares process,
network, filesystem, resource, and secret isolation. PDECert has no local
`exec`, shell, or subprocess fallback.

Successful isolated execution is not a certificate. The bounded JSON output is
materialized through the restricted symbolic parser and then evaluated by the
ordinary verifier. PDECert currently ships the contract and disabled default,
not a production sandbox backend. See the
[generated-program guide](docs/generated-programs.md) and
[isolation decision](docs/adr/0005-generated-program-isolation.md).

## Current limits

The prototype does not yet define weak or viscosity solution semantics. It has
no built-in validated numerical backend, and callable sampled passes remain
`INCONCLUSIVE`. The CLI has a structural input-operation budget, but
intermediate symbolic expressions still need enforceable memory limits.
Stronger multivariate domain analysis and supported a posteriori solution-error
bounds are also missing. Read the full
[limitations and threats-to-validity statement](LIMITATIONS_AND_THREATS_TO_VALIDITY.md)
before interpreting a verifier or benchmark result.

The next milestones are tracked in [ROADMAP.md](ROADMAP.md). The
[architecture map](ARCHITECTURE.md) explains the package layers and contributor
workstreams. Contributions that follow the focused workflow in
[CONTRIBUTING.md](CONTRIBUTING.md) are welcome.

## Citation

If PDECert or the pilot benchmark supports your work, cite the software using
[`CITATION.cff`](CITATION.cff). GitHub can render that file as APA or BibTeX.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
