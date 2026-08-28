# Contributing

Thanks for taking an interest in PDECert. The project is still small, so focused
changes are easiest to review.

Start with [`ARCHITECTURE.md`](ARCHITECTURE.md) to see the package layers,
extension boundaries, evidence rules, and six contributor workstreams. New work
should deliver one vertical slice rather than changing several unrelated layers.

## Choose a workstream

| Workstream | Good first contribution | Extra review requirement |
| --- | --- | --- |
| Soundness core | Improve an incomplete diagnostic or resource limit | Adversarial tests and compatibility analysis |
| Artifact representations | Validate metadata for one candidate representation | Demonstrate faithful round-trip or evaluation semantics |
| Verification backends | Add one narrowly supported checker | State assumptions, evidence strength, and abstention boundary |
| Benchmark science | Contribute one natural failure or matched case | Preserve provenance and follow independent review protocol |
| Ecosystem integrations | Add an optional loader or report adapter | Test without making the integration a core dependency |
| Developer experience | Improve setup, CI, documentation, or packaging | Tie the change to a reproducible contributor problem |

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

### Definition of done

A behavior-changing pull request is complete when it includes:

- a concise statement of supported and unsupported scope;
- implementation behind the appropriate extension boundary;
- valid, invalid, and unsupported or inconclusive tests;
- one minimal example, fixture, or replayable witness;
- updated user or contributor documentation;
- exact verification commands and results; and
- a changelog entry when public behavior or a report contract changes.

Do not split one coherent capability into activity-only pull requests. Large
backends should instead be decomposed into independently useful vertical slices,
such as representation validation, derivative reconstruction, and convergence
evaluation.

Candidate-corpus records must retain the unedited generator output and complete
origin metadata. Do not reconstruct or polish an output and present it as a raw
sample. New records should remain `pending` until a person applies the published
labeling protocol.
Validate a versioned corpus before requesting review:

```bash
pdecert corpus validate path/to/corpus.json
pdecert corpus validate path/to/atlas-directory
```

Do not import `results/provisional-review.json` as a human review. Complete the
blind pass in `corpus/LABELING.md` first, use a public reviewer identifier, and
retain disagreement notes when a second reviewer is needed.

For soundness-sensitive changes, a new check must not turn finite numerical
sampling into a `PROVED` result. If the reasoning is incomplete, return
`INCONCLUSIVE` and record why.

Every new checker or evaluator must document:

1. accepted artifact and problem representations;
2. obligations and solution semantics it addresses;
3. whether its evidence is exact symbolic, rigorously bounded, or empirical;
4. the form of a replayable refutation witness;
5. unsupported inputs and abstention behavior; and
6. dependencies, precision, tolerances, seeds, and resource limits needed for
   reproduction.

Symbolic complexity limits must preserve counterexample search where practical.
Skipping an over-budget exact check is an incomplete proof attempt, not evidence
that a candidate is valid or invalid.

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

Portable operator additions must also follow the declared subset in
[`ADR-0006`](docs/adr/0006-portable-operator-lowering.md). Add one symbolic and
one callable regression for any new lowering, plus an unsupported test when the
same syntax has backend-dependent semantics. Do not silently approximate a
symbolic function with a numerically different PyTorch operation.

Reusable tasks follow the candidate-free boundary in
[`ADR-0007`](docs/adr/0007-candidate-free-problem-templates.md). Put trusted
operators and solution semantics in a `ProblemTemplate`; bind model or solver
outputs separately. A new template-semantics value requires a faithful backend
and its own architectural review. Do not label weak, entropy, or distributional
problems as `classical_strong` to make them pass version-1 validation.

Published evaluation runs should use the digest-bound contract in
[`ADR-0008`](docs/adr/0008-digest-bound-run-manifests.md). Keep raw candidate
artifacts, reports, and templates as separate files; do not edit a candidate to
make its digest agree with a desired outcome. A digest establishes content
identity only, so descriptions must not present manifest validation as proof,
authorship, or trusted execution.

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
