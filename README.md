<p align="center">
  <img src="assets/pdecert-icon.png" alt="PDECert project icon" width="200">
</p>

<h1 align="center">PDECert</h1>

<p align="center"><strong>Check a PDE candidate against its equation and conditions.</strong></p>

PDECert checks symbolic expressions and pointwise PyTorch fields. It reports
what the checks establish, where a candidate fails, and what remains undecided.
Use it to audit a solver output, check an LLM proposal, or compare symbolic and
neural candidates for the same problem.

This is an early research package, not a general PDE solver or theorem prover.
Its three outcomes have different meanings:

| Result | Meaning |
| --- | --- |
| `PROVED` | Exact symbolic checks establish the encoded obligations and supported domain checks. This is not a proof-assistant certificate or an existence/uniqueness theorem. |
| `REFUTED` | A check finds a violation, with a witness and its evidence type. A floating-point counterexample is empirical, not an exact proof. |
| `INCONCLUSIVE` | The checks cannot decide. This includes passing numerical samples and unsuccessful symbolic simplification. |

**Passing samples never becomes proof.** Reports keep exact symbolic evidence
separate from empirical diagnostics. The format reserves `RIGOROUS_BOUND` for
validated bounds, but no built-in backend currently produces them.
Read the [report format](docs/evidence-reports.md) and
[limitations](LIMITATIONS_AND_THREATS_TO_VALIDITY.md) before using a result as
scientific evidence.

## Five-minute quickstart

In a Python 3.10–3.14 environment:

```bash
python -m pip install pdecert==0.1.1rc2
pdecert quickstart
pdecert quickstart --json > pdecert-quickstart.json
```

After installation, this demo runs offline with no credentials or optional
dependencies. It shows an exact symbolic result, a condition violation, a
sampled pass that remains inconclusive, and a recorded agent rejection-and-repair
trace. The trace uses fixed examples; it does not call a live model.

Next, [check your own expression](docs/check-your-candidate.md). The guide
includes a complete problem, one candidate line to change, and examples of all
three outcomes. See the [quickstart guide](docs/quickstart.md) for expected output.

## What can I use it for?

| Task | Start here |
| --- | --- |
| Check a closed-form solution against a fixed problem | [Own-candidate guide](docs/check-your-candidate.md) |
| Check a pointwise PyTorch function or PINN | [Callable example](examples/autodiff_heat.py) |
| Use the same equation and conditions for both representations | [Problem templates](docs/problem-templates.md) |
| Send reports and counterexamples back to an agent | [Agent integration](docs/agent-integration.md) |
| Evaluate a corpus without mixing evidence or pending labels | [Atlas evaluation](docs/atlas-evaluation.md) |
| Add a checker or comparison method | [Architecture](ARCHITECTURE.md) and [baseline adapters](docs/baseline-adapters.md) |

The installed release and the development branch are not interchangeable.
The quickstart above uses the published release candidate. Repository examples,
corpora, and experiment commands below require a checkout; `main` may contain
changes not yet on PyPI. See [CHANGELOG.md](CHANGELOG.md).

## Install and run

For development and the repository examples:

```bash
git clone https://github.com/oroikono/PDECert.git
cd PDECert
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
pdecert verify examples/exact_heat.json --output report.json
```

The `verify` command exits with `0` for `PROVED`, `1` for `REFUTED`,
`2` for `INCONCLUSIVE`, and `64` for invalid input. A nonzero exit code is
not necessarily a software error: read the report.

For a problem already saved as JSON:

```python
from pdecert import load_case, verify

case = load_case("examples/exact_heat.json")
report = verify(case.problem, case.candidate_fields, symbolic_timeout=2.0)
print(report.status)
print(report.to_dict())
```

When evaluating generated candidates, keep the problem outside the generator's
control. A [problem template](docs/problem-templates.md) defines the variables,
domain, equation, and conditions; `bind_symbolic_candidate` supplies only the
proposed fields. The [own-candidate guide](docs/check-your-candidate.md) shows
this without depending on repository files.

Expression strings use a restricted arithmetic grammar, not arbitrary Python.
The CLI limits individual symbolic checks to two seconds and 10,000 input
operations by default. These are not total runtime or memory limits. Untrusted
service workloads still need process and resource isolation.

## PyTorch and trained candidates

Install the optional backend from the checkout:

```bash
python -m pip install -e ".[dev,autodiff]"
python -m examples.autodiff_heat
python -m examples.problem_template
python -m experiments.trained_burgers_pair
```

`CallableCandidate` accepts named functions or modules with input shape
`(points, variables)`. Each output row must depend only on the corresponding
input row. Cross-sample attention and training-mode batch operations do not
provide pointwise derivatives under this checker.

The Burgers and Fisher–KPP examples pair an exact symbolic traveling wave with
a separately trained, frozen PINN. They are small reproducible fixtures, not
evidence that arbitrary PINNs can be certified. Their reports stay separate:
a symbolic proof does not transfer to the network, and low training loss does
not guarantee a held-out residual will pass. See
[trained pairs and replay requirements](docs/trained-callable-pairs.md) and the
[frozen-callable format](docs/frozen-callables.md).

## LLM and agent workflows

An agent can propose a candidate, receive a report, and submit a linked repair.
PDECert retains the original response and keeps the trusted problem separate
from the proposal. It supports a framework-free loop and an optional smolagents
integration:

```bash
python -m examples.agent_repair_loop
```

The [integration guide](docs/agent-integration.md) explains the tool interface
and live-run setup. There is also one
[recorded hosted-model smoke run](results/agent-smoke/README.md); it demonstrates
an integration, not broad agent reliability.

`ProgramCandidate` can record generated source, but does not execute it.
Execution requires an explicitly supplied sandbox. PDECert ships the interface
and a disabled default, **not a production sandbox**.
See [generated programs](docs/generated-programs.md).

## PDE Failure Atlas

The [Atlas](corpus/ATLAS.md) stores candidates alongside their stated problems,
original outputs, provenance, review status, and machine reports. It separates
machine proposals from human labels.

From a checkout:

```bash
pdecert corpus validate corpus/pilot.json
pdecert corpus validate corpus/community
pdecert corpus validate corpus/matched
```

The labeled pilot has 20 records: 10 SymPy outputs and 10 local Qwen3-0.6B
generations. It has one named reviewer. All SymPy records are labeled valid and
all Qwen records invalid, so origin and verdict are confounded. This pilot
cannot establish performance on new solvers or models.

The [published pilot report](results/pilot-benchmark.json) gives PDECert and
full-condition collocation the same 20/20 classification result. Direct SymPy
simplification decides 13/20 and abstains on seven. These are descriptive pilot
results, not evidence of general superiority. See the
[collection and review notes](corpus/README.md).

The pilot is available on
[Hugging Face](https://huggingface.co/datasets/oroikono/pdecert-pilot) at the
[immutable first-release revision](https://huggingface.co/datasets/oroikono/pdecert-pilot/commit/db690f9b161762ea288dd5dfb4b6b2f999c48e03).
The [matched preview](corpus/matched/README.md) has a symbolic proposal and a
trained PINN for one Fisher–KPP problem. Both remain pending review; evaluator
outputs are not independent ground truth.

To reproduce the labeled pilot comparison:

```bash
python -m experiments.run_benchmark corpus/pilot.json \
  --output /tmp/pdecert-pilot-benchmark.json
```

For newer records, see [Atlas evaluation](docs/atlas-evaluation.md),
[comparison baselines](docs/baseline-adapters.md), and the
[blind labeling protocol](corpus/LABELING.md). File hashes detect changed
content; they do not prove authorship, correct execution, or a correct label.
[Run manifests](docs/run-manifests.md) record what an evaluation used.

## Limits and next steps

Current checks target classical, pointwise problems with supported expressions
and domains. There are no weak- or viscosity-solution guarantees, gridded
neural-operator backend, validated numerical certification backend, or general
solution-error bounds. Symbolic domain analysis is incomplete, and intermediate
expression growth is not memory-bounded.

The next priority is a small, independently reproducible cross-artifact
benchmark and a reliable contributor workflow—not a larger list of advertised
features. [ROADMAP.md](ROADMAP.md) defines the release gates.
The [research landscape](docs/research-landscape.md) credits related work and
explains which novelty claims the project does not make.

## Contributing

A useful first contribution is one reproducible failure, a regression test, a
clearer diagnostic, or an optional adapter that solves a real user's problem.
Use the [failure-case form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml)
to share an unchanged candidate and its conditions. You do not need to learn
the corpus format first.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, review, and evidence rules.
Keep changes focused and preserve raw outputs and frozen benchmark records.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software. PDECert is
[MIT licensed](LICENSE).
