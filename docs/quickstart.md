# Quickstart

Run a small heat-equation example to see a proof, a counterexample, and an
inconclusive result. You need Python 3.10–3.14. Paste these commands into a
terminal on macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pdecert==0.1.1rc2
pdecert quickstart
```

Installation needs network access. After that, the demonstration runs offline
using the core package, with no repository checkout, benchmark download, or
model account.

## What to expect

The command checks three symbolic expressions against the same equation,
initial data, and boundary data:

| Candidate | Result | Evidence |
| --- | --- | --- |
| Exact solution | `PROVED` | `EXACT`: the encoded equation, conditions, and domain checks hold symbolically. |
| Correct PDE, wrong initial and right boundary data | `REFUTED` | `EMPIRICAL`: a sampled point shows a violation you can replay. |
| Nonzero residual below the numerical tolerance | `INCONCLUSIVE` | Passing the sampled checks does not prove correctness. |

Next, [check your own symbolic candidate](check-your-candidate.md). That guide
provides a complete example and shows which expression to change.

## Save the full reports

```bash
pdecert quickstart --json > pdecert-quickstart.json
```

The JSON includes each report, its evidence and counterexample, and the
evaluation settings. It also records two linked proposals: a rejected
`attempt-1` and a repaired `attempt-2` that is proved. These are fixed examples;
the command does not run a live language model. The trace keeps proposal
provenance separate from verifier evidence and includes proposal hashes and
the parent link. Raw proposal text is excluded from the default JSON trace.

Exit code `0` means the demonstration reproduced all expected results. Exit
code `70` means an expected outcome changed. The JSON also contains these
self-checks, so the command can serve as an installation check.

These examples do not measure broad PDE coverage or model quality. The exact
result applies only to the encoded classical strong-form checks; the sampled
pass is not a certificate. Read the
[limitations](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md) before using a report
in an evaluation or publication.
