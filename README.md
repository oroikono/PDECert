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

- `PROVED` when every current obligation is an exact symbolic identity;
- `REFUTED` when it finds a singularity or a numerical counterexample;
- `INCONCLUSIVE` when the available checks cannot decide.

Numerical sampling is only used to refute. Passing sampled points is never
treated as a proof.

> [!IMPORTANT]
> This is a research prototype, not a general theorem prover. The current
> `PROVED` result applies only to symbolic obligations and domain checks
> represented by the input problem. Passing callable samples is always
> `INCONCLUSIVE`; sampling can refute but cannot prove.

## Why this exists

PDE solvers and language models can produce expressions that look convincing
and have a small residual on a fixed grid. That does not guarantee that they
satisfy the full problem. A candidate can fail at a boundary, hide a pole
between collocation points, or work only at one parameter value.

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
```

The installed command accepts one versioned JSON case and prints a stable JSON
report:

```bash
pdecert verify examples/exact_heat.json
pdecert verify examples/exact_heat.json --output report.json
pdecert verify examples/exact_heat.json --symbolic-timeout 5
```

Exit code `0` means `PROVED`, `1` means `REFUTED`, `2` means `INCONCLUSIVE`,
and `64` reports an unreadable or invalid input file. A non-zero result is not
automatically a software failure; consumers should also read `report.status`.
The CLI gives each singularity and symbolic-identity check two seconds by
default. A deadline, unsupported deadline environment, or undecidable symbolic
operation is recorded under `report.incomplete_reasons` and cannot produce a
`PROVED` result.

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
problem and requires a concrete witness for refutation.

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

## Current limits

The prototype does not yet define weak or viscosity solution semantics. It also
needs operation and memory budgets, stronger multivariate domain analysis,
interval arithmetic, and supported a posteriori error bounds. Real-time symbolic
deadlines are currently available only from the main thread on platforms that
provide interval timers. When a check is incomplete, the intended behavior is
`INCONCLUSIVE`.

The next milestones are tracked in [ROADMAP.md](ROADMAP.md). Contributions that
add one focused capability together with tests are welcome.

## Citation

If PDECert or the pilot benchmark supports your work, cite the software using
[`CITATION.cff`](CITATION.cff). GitHub can render that file as APA or BibTeX.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
