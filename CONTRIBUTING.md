# Contributing

Thanks for taking an interest in PDECert. The project is still small, so focused
changes are easiest to review.

## Good ways to begin

- Share a real symbolic PDE candidate that was accepted by a residual or
  collocation check but fails the stated problem.
- Add one PDE example with its equation, domain, conditions, expected outcome,
  and a regression test.
- Improve an `INCONCLUSIVE` diagnostic without turning numerical sampling into
  proof.
- Connect the JSON report to a symbolic solver or scientific-agent workflow.

If you are unsure whether a case fits, open an issue first. Include the equation,
domain, initial or boundary conditions, candidate expression, expected solution
semantics, and how the candidate was produced. Small reproducible cases are the
most useful.

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
Do not import `results/provisional-review.json` as a human review. Complete the
blind pass in `corpus/LABELING.md` first, use a public reviewer identifier, and
retain disagreement notes when a second reviewer is needed.

For soundness-sensitive changes, a new check must not turn finite numerical
sampling into a `PROVED` result. If the reasoning is incomplete, return
`INCONCLUSIVE` and record why.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
python -m pytest
```
