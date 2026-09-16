# Historical source identity and current evaluation

Frozen training provenance and a new evaluator are different records. The
Fisher--KPP history binds 18 exact source/input files. These hashes continue to
identify the original bytes even when the current package changes.

## Three different operations

| Operation | What is checked | What it does not establish |
| --- | --- | --- |
| Historical content validation | Frozen artifact identity and all 18 historical file hashes | That training occurred or that a report was numerically reproduced |
| Current evaluation | The same frozen candidate and problem evaluated with separately bound current source and configuration | Historical evaluator replay, independent reproduction or a new human label |
| Historical numerical replay | Explicit execution in a complete matching historical checkout and recorded environment | Platform-independent behavior or reproduction of training merely from weights |

The small [source snapshot](../benchmarks/historical/fisher-kpp-source-v1/README.md)
is benchmark data. It is never imported or executed by the receipt validator.
Its supplying revision `c962d9c2df4a37bd682fd187e739c8db0edfb1d5` has matching
source bytes; it is not represented as the original training commit. The frozen
artifact, integrity records and results remain unchanged.

## Inspect and validate without PyTorch

From the repository root, with the package installed from this checkout:
relative input paths use the current directory; `--repository-root` validates
the source boundary and does not change that working directory.

```bash
python -m experiments.current_fisher_kpp_pair \
  --historical-source-root benchmarks/historical/fisher-kpp-source-v1 \
  --output /tmp/fisher-kpp-source-inspection.json
python -m experiments.current_fisher_kpp_pair \
  --historical-source-root benchmarks/historical/fisher-kpp-source-v1 \
  --validate /tmp/fisher-kpp-source-inspection.json
```

Use a fresh output path: existing files are refused. The inspection records
`operation: content_identity_check`, no report, and `historical_replay: false`.
Validation of a saved receipt checks its file and report bindings without
recomputing numerical diagnostics. An output hash is content identity, not a
semantic check of a supplied report, an execution attestation or evidence that
an independent party reproduced it.

The underlying Python API accepts any explicitly supplied matching source root:

```python
from pdecert import validate_frozen_callable_integrity

validate_frozen_callable_integrity(
    "benchmarks/matched/fisher-kpp-classical-01/pinn.json",
    "benchmarks/matched/fisher-kpp-classical-01/integrity.json",
    repository_root=".",
    historical_source_root="benchmarks/historical/fisher-kpp-source-v1",
)
```

Omitting `historical_source_root` retains strict validation against the current
checkout. It correctly fails if a historically bound source has changed; there
is no automatic fallback, hash exclusion, remote fetch or code execution.

## Produce new current diagnostics

Install the optional autodiff dependency in your evaluation environment, then
explicitly request numerical evaluation:

```bash
python -m experiments.current_fisher_kpp_pair \
  --historical-source-root benchmarks/historical/fisher-kpp-source-v1 \
  --evaluate --output /tmp/fisher-kpp-current-evaluation.json
```

This evaluates the preserved frozen weights; it does not retrain the PINN or
call a generative model. It retains separate symbolic and callable reports,
without an overall status. Tolerances are `1e-9` and `1e-3`, respectively,
with six samples per axis and a two-second symbolic deadline. The callable
uses CPU float64. Passing samples remain `INCONCLUSIVE`; a valid finite witness
can give an empirical `REFUTED` result.

A current source receipt hashes all `src/pdecert/**/*.py`, explicit runner
sources, its schema and `pyproject.toml`. It records local Git HEAD and tracked
patch identity where available; untracked source filenames are explicit and
their bytes are covered by the source map. Git data may be unavailable in source
archives. Python/platform and package versions are recorded, and the absence of
a dependency lock is explicit. External dependency bytes and in-memory monkey
patches are outside this content-only contract. The runner checks source origins
and pre/post file hashes but is not a security sandbox or an attestation service.

## Execute historical code explicitly

For historical numerical replay, use a separate complete trusted checkout with
all bound files matching the integrity record. A local Git object can supply a
matching checkout without any download:

```bash
history_dir="$(mktemp -d)"
git archive c962d9c2df4a37bd682fd187e739c8db0edfb1d5 | tar -x -C "$history_dir"
cd "$history_dir"
# Use a separately provisioned environment with the recorded dependency versions.
PYTHONPATH=src python -m experiments.trained_fisher_kpp_pair \
  --output /tmp/fisher-kpp-historical-replay.json
```

This command explicitly runs trusted repository code, never candidate-supplied
code. The archive revision is a source of matching bytes, not an assertion that
it was the training HEAD. Compare complete reports and environments; do not
expect platform-independent floats or claim bitwise training regeneration.
The 18-file data snapshot alone is insufficient for execution because it omits
unbound package/runtime files. See [ADR-0011](adr/0011-historical-source-and-current-evaluation.md).
