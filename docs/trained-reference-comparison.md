# Prediction error beside PDE and boundary diagnostics

This walkthrough evaluates the existing frozen Fisher--KPP PINN. It connects
the [reference-field API](reference-fields.md) to actual PyTorch model outputs
and checks each represented PDE, initial, and boundary obligation separately.
No training or model-provider call is needed.

## Run from a checkout

```bash
python -m pip install -e ".[dev,autodiff]"
python -m experiments.trained_fisher_kpp_reference > trained-reference.json
python -m pytest tests/test_trained_fisher_kpp_reference.py
```

The command requires PyTorch but adds no new package dependency. It resolves the
bundled files relative to its checkout, not the shell's current directory. An
explicit `--repository-root /path/to/PDECert` can select a complete copy of the
same inputs. The guide and experiment are repository resources, not installed
CLI commands in the published release candidate.

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
configuration, and retained source-file digests. It also checks that the actually
imported evaluator modules match the fixture's declared sources and records
digests of the new runner and comparison implementation. Changed or missing
required bindings are errors, not a silently accepted new baseline.

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
