# Pilot benchmark release

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

## 5. Publish only the inspected bundle

```bash
hf auth whoami
hf repos create oroikono/pdecert-pilot --repo-type dataset --public --exist-ok
hf upload oroikono/pdecert-pilot dist/pdecert-pilot . \
  --repo-type dataset \
  --commit-message "Publish the labeled PDECert pilot"
```

After upload, verify that the Hub renders the dataset card, exposes exactly 20
rows in the `test` split, and shows the same corpus digest as the local manifest.
Then link the immutable Hub commit from the GitHub release notes.
