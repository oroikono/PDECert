# Prediction error beside PDE and boundary diagnostics

This walkthrough evaluates the existing frozen Fisher--KPP PINN. It connects
the [reference-field API](reference-fields.md) to actual PyTorch model outputs
and checks each represented PDE, initial, and boundary obligation separately.
No training or model-provider call is needed.

## Run from a checkout

```bash
python -m pip install -e ".[dev,autodiff]"
python -m experiments.trained_fisher_kpp_reference \
  --historical-source-root benchmarks/historical/fisher-kpp-source-v1 \
  > trained-reference-current.json
python -m pytest tests/test_trained_fisher_kpp_reference.py
```

The command requires PyTorch but adds no new package dependency. It resolves the
bundled files relative to its checkout, not the shell's current directory. An
explicit `--repository-root /path/to/PDECert` must identify the checkout from
which the package and runner are imported. Supply an absolute historical source
path when running outside the checkout. The guide and experiment are repository
resources, not installed CLI commands in the published release candidate.

Accepted scope is the represented classical Fisher--KPP problem
`u_t - u_xx - u*(1-u) = 0` on `x in [-6,6]`, `t in [0,2]`, with its three
declared traces and the restricted CPU float64 dense-tanh artifact. The reference
is the analytical traveling-front formula
`(1+exp(x/sqrt(6)-5*t/6))**(-2)`, evaluated using Python binary64 arithmetic.
The template digest is pinned so the formula cannot silently be compared with a
different equation, domain, or boundary specification.

## What the report shows

Representative local results with Python 3.12.13 and PyTorch 2.13.0:

| Region | Rows | Prediction RMSE against reference | Maximum sampled obligation residual | Diagnostic at tolerance 0.001 |
| --- | ---: | ---: | ---: | --- |
| Interior | 36 | 0.00340621 | 0.00912092 | `REFUTED`, empirical |
| Initial condition | 6 | 0.00087438 | 0.00137304 | `REFUTED`, empirical |
| Left boundary | 6 | 0.00060620 | 0.00088834 | `INCONCLUSIVE`, observed pass |
| Right boundary | 6 | 0.00021302 | 0.00033269 | `INCONCLUSIVE`, observed pass |

Prediction error and PDE residual are different quantities, not interchangeable
measures of correctness. The table is a description of one small trained
network, not an accuracy estimate for PINNs generally. Values may vary slightly
with platform and numerical libraries. The sampled failures exceed the selected
tolerance; they are not interval-certified counterexamples.

The standard callable verifier can stop after its first refutation. This
experiment invokes it once per obligation so an interior failure cannot prevent
initial-condition or boundary diagnostics from running. Each row contains:

- `samples`: every coordinate, prediction, and reference value used by its
  reference comparison;
- `reference_comparison`: empirical per-field metrics and normalized-input
  identity under the existing version-1 metric schema;
- `diagnostic_report`: a fresh native evidence report, including a witness or
  an inconclusive reason;
- `source_obligation_id`: the obligation's ID in the original template.

Native diagnostic IDs are local to each single-obligation problem and therefore
use `constraint:0`. The enclosing source ID and name identify the original
obligation. There is no combined verdict or relabeling of the corpus.

## Reproduction and integrity

Each unfixed coordinate uses the interval fractions
`[0.113, 0.271, 0.419, 0.613, 0.787, 0.937]`, matching the existing six-point
autodiff sampler. Cartesian rows have `x` outermost and `t` innermost. Initial
and boundary sets instead fix their declared surface coordinate. The sample
values are included so that metric calculations and witnesses can be replayed.

Before materialization the experiment verifies the frozen artifact, weights,
configuration, and all 18 retained historical source-file digests in the explicit
historical source root. The original integrity record is itself digest-pinned,
so removing or replacing historical bindings fails validation. Archived files
are checked as data and are never imported by this runner.

The `trained-fisher-kpp-reference-v2` report retains that historical record in
`integrity`. Its `current_evaluator` source receipt separately hashes every
current package Python file, the runner, relevant schemas and `pyproject.toml`.
`active_inputs` binds the exact fixture, integrity record and template used.
This replaces the version-1 `source_digests` map, which required current evaluator
bytes to equal historical bytes. The per-obligation numerical reports and the
reference-comparison version-1 schema keep their existing meanings.

The runner checks imported package/runner origins and repeats input and source
checks after evaluation. Reports state `historical_replay: false` and
`integrity_scope: content_identity_only`: they describe current evaluation of
preserved weights. Source hashes do not attest in-memory execution or bind all
external dependency bytes. See [source replay](source-replay.md) for the separate
historical replay procedure and the snapshot's limits.

Inspect the same historical and current bindings without PyTorch or evaluation:

```python
from experiments.trained_fisher_kpp_reference import inspect_inputs

provenance = inspect_inputs(
    historical_source_root="benchmarks/historical/fisher-kpp-source-v1",
)
print(provenance["current_evaluator"])
```

The Python `run()` entrypoint now requires `historical_source_root` explicitly.
`load_inputs()` accepts the same keyword; omitting it retains strict historical
validation against the current checkout. Provenance tests run in the core-only
environment; only numerical tests require PyTorch.

The output retains training provenance, all integrity fields, execution versions,
dtype, device, thread count, sampling fractions, and tolerance. Stored training
losses are historical metadata, not fresh training results. No weights, model
responses, annotations, or old result files are rewritten. Hashes establish
identity and consistency, not independent evidence that the training occurred.

## Limits and next integration

This is a trained-fixture consumer of the public API, not an arbitrary checkpoint
loader or the Atlas reference-field adapter. It does not support other problem
templates, irregular domains, weak solutions, or a gridded/neural-operator lane.
The reference formula's rounding error is unvalidated; zero sampled discrepancy
would not certify the reference, network, or continuous-domain solution error.
Passing residual samples remain `INCONCLUSIVE`.

The PINN was separately trained using PDE and trace targets, but its traces cue
the analytical reference. This demo is neither an independent human label nor
an external reproduction. The open-model/callable Atlas records remain pending.
Tests also use an analytical callable as an explicit positive control; that
control is not substituted for the trained model in the example.

The next integration should reuse this comparison interface with a documented
external solver/model loader and retain its native inference and reference
uncertainty, rather than claiming that one fixture completes the baseline suite.
