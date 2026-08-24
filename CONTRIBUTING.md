# Contributing

Thanks for taking an interest in PDECert. The project is still small, so focused
changes are easiest to review.

## Good ways to begin

- Share a real symbolic, neural, or numerical PDE candidate that was accepted by
  a residual or collocation check but fails the stated problem.
- Add one PDE example with its equation, domain, conditions, expected outcome,
  and a regression test.
- Improve an `INCONCLUSIVE` diagnostic without turning numerical sampling into
  proof.
- Connect the JSON report to a symbolic solver or scientific-agent workflow.

If you are unsure whether a case fits, use the
[failure-case issue form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml).
It collects the equation, domain, conditions, original artifact, provenance,
expected semantics, and redistribution terms without requiring corpus JSON.
Small reproducible cases are the most useful. The atlas scope, record lifecycle,
and planned artifact lanes are documented in [`corpus/ATLAS.md`](corpus/ATLAS.md).

## Pull requests

Before opening a pull request:

1. Explain the failure mode or verifier capability in an issue.
2. Keep the pull request to one main behavior.
3. Add a regression test or a minimal reproducible example.
4. Run `pytest` and `ruff check .` locally.
5. State what the result proves, and what it does not prove.

Candidate-corpus records must retain the unedited generator output and complete
origin metadata. Do not reconstruct or polish an output and present it as a raw
sample. New records should remain `pending` until a person applies the published
labeling protocol.
Validate a versioned corpus before requesting review:

```bash
pdecert corpus validate path/to/corpus.json
```

Do not import `results/provisional-review.json` as a human review. Complete the
blind pass in `corpus/LABELING.md` first, use a public reviewer identifier, and
retain disagreement notes when a second reviewer is needed.

For soundness-sensitive changes, a new check must not turn finite numerical
sampling into a `PROVED` result. If the reasoning is incomplete, return
`INCONCLUSIVE` and record why.

## Checker extensions

New verification techniques should implement the public `Checker` protocol
rather than adding a branch to `verify()`. Start from a fresh
`default_checker_registry()` and add the checker explicitly so the execution
environment remains reproducible. A checker may only prove obligation IDs from
its `CheckContext`; a refutation must include a concrete `Witness`.

Include tests for supported, unsupported, and inconclusive inputs. State the
mathematical assumptions under which proof evidence is sound. The architecture
and trade-offs are recorded in
[`docs/adr/0001-plugin-first-extension-architecture.md`](docs/adr/0001-plugin-first-extension-architecture.md).

Callable candidates follow the representation boundary described in
[`ADR-0002`](docs/adr/0002-general-solution-artifacts.md). PyTorch is optional:
symbolic contributors should not need it. Autodiff checks must cover the PDE and
represented initial or boundary surfaces, report non-finite outputs, and remain
`INCONCLUSIVE` after finite sampled success.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
python -m pytest
```

For callable and PINN-related changes, also run:

```bash
pip install -e ".[dev,autodiff]"
python -m pytest tests/test_autodiff.py
python -m examples.autodiff_heat
```
