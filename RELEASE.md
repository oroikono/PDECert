# Release procedures

## Python package release candidate

PDECert publishes with PyPI Trusted Publishing. The release workflow builds
from the GitHub release tag, checks both distributions, and exchanges GitHub's
short-lived OpenID Connect identity for a PyPI upload token. No long-lived PyPI
token belongs in the repository or GitHub secrets.

Before the first publication, create a pending publisher for project name
`pdecert` in PyPI with these exact values:

- owner: `oroikono`
- repository: `PDECert`
- workflow: `release.yml`
- environment: `pypi`

Create the matching GitHub `pypi` environment and require manual approval for
deployment. See PyPI's
[trusted-publisher guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

Before publishing:

1. Confirm `pyproject.toml`, `CITATION.cff`, and the release tag name the same
   version. Python version `0.1.1rc1` uses Git tag `v0.1.1rc1`.
2. Confirm the pull-request and `main` test matrices are green.
3. Build twice with `python -m build` and byte-compare the wheel and sdist.
4. Run `python -m twine check dist/*`.
5. Install the wheel and sdist into separate empty environments. Run
   `pdecert --help` and verify `examples/exact_heat.json` from outside the
   repository checkout.
6. Publish a GitHub prerelease from the exact tag. Approve the protected `pypi`
   deployment only after the build job succeeds.

After publication, install from PyPI in a new environment, compare PyPI's file
digests with the inspected artifacts, and rerun the installed command. A broken
release is yanked and superseded; published files and tags are never replaced.

## Pilot benchmark release

This checklist prevents an unlabeled, mismatched, or synthetic test fixture from
being presented as the public pilot benchmark.

## 1. Finish the independent review

```bash
python -m experiments.review_corpus
```

Confirm that the runner reports `20/20`. Compare the completed private review
with `results/provisional-review.json` only after the blind pass. Recompute any
disagreement and follow `corpus/LABELING.md` when a second reviewer is needed.

## 2. Import and inspect labels

```bash
python -m experiments.apply_review private-reviews/pilot-review.json \
  --annotator oroikono \
  --confirm-independent-review \
  --output corpus/pilot-labeled.json
```

Inspect every annotation diff before replacing `corpus/pilot.json`. Do not use
the confirmation flag unless the independent review actually happened.

## 3. Run all gates and benchmark methods

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m experiments.run_benchmark corpus/pilot.json \
  --output results/pilot-benchmark.json
```

Read the per-record outcomes and the timing note. A small pilot result is
descriptive evidence, not a population-level performance claim.

## 4. Build and inspect the Hub bundle

```bash
python -m experiments.build_release corpus/pilot.json \
  --benchmark results/pilot-benchmark.json \
  --output dist/pdecert-pilot
```

The builder refuses pending annotations, digest mismatches, altered truth rows,
recomputed-metric mismatches, non-standard JSON, and nonempty destinations.
Inspect `README.md`, `data/pilot.jsonl`, `results/pilot-benchmark.json`, and
`manifest.json` in the output directory.

## 5. Stage only the inspected bundle

```bash
hf auth whoami
hf repos create oroikono/pdecert-pilot --repo-type dataset --private --exist-ok
hf upload oroikono/pdecert-pilot dist/pdecert-pilot . \
  --repo-type dataset \
  --commit-message "Publish the labeled PDECert pilot"
```

Download every staged file from its immutable Hub revision and compare its hash
with the local bundle. Confirm that no private review file, credential, or local
path is present.

## 6. Make the verified revision public

Before changing visibility, explicitly approve the public payload and its
destination. The public release contains 20 labeled candidate records, raw
model or solver outputs and provenance, the public reviewer identifier, the
dataset card, the benchmark report, and the checksum manifest.

```bash
hf repos settings oroikono/pdecert-pilot --repo-type dataset --public
```

After the visibility change, verify that the Hub renders the dataset card,
exposes exactly 20 rows in the `test` split, and shows the same corpus digest as
the local manifest. Then link the immutable Hub commit from the GitHub release
notes.

## First public release

- Dataset: <https://huggingface.co/datasets/oroikono/pdecert-pilot>
- Immutable revision:
  `db690f9b161762ea288dd5dfb4b6b2f999c48e03`
- Corpus SHA-256:
  `4be9178edd30fcc561f21e83375713f4b38338484d75a0c8a7c8088e9c4369fb`
- Public audit: 20 labeled rows; all four release files byte-identical to the
  privately reviewed bundle.
