# Hugging Face pilot consumer check

This optional engineering check exercises the actual Hugging Face `datasets`
library, not just JSON decoding or the Dataset Viewer. It checks transport
fidelity before running PDECert. No inference, training, new labels, Hub upload,
or credentials are needed. The core package dependencies are unchanged.

## Reproduce

From a checkout of the commit you want to test, use a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]" -r requirements/hub-integration.txt
python -m pip check
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q tests/test_hub_pilot_smoke.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m experiments.hub_pilot_smoke --output local-smoke.json
python -m experiments.hub_pilot_smoke --from-hub --output hub-smoke.json
```

The first smoke run builds the release JSONL from committed pilot inputs and
loads it through the installed JSON/Arrow implementation with network access
disabled. The second reads the public Hub anonymously at the first immutable
pilot commit, with the published `default` configuration and `test` split.
The JSONL download and prepared Arrow data use new temporary caches; Hub card
metadata may use the library's normal cache. Output files must not already exist.
Local mode also temporarily sets the pinned libraries' active offline switches
and restores them on success or failure, so Datasets' optional download-counter
request cannot run. This process-global guard is for the single-threaded smoke
command, not a concurrent service or a network-security sandbox.

The optional dependency lock pins Datasets 5.0.1, Hub 1.27.0, PyArrow 25.0.1 and
their resolved dependencies with Python/platform markers. CI exercises Python
3.10 and 3.14; this is not a claim to support every older Datasets version.
To deliberately update the lock, edit `requirements/hub-integration.in`, run
the `uv pip compile` command in the generated file's header, and repeat both
consumer checks. Record the tested PDECert Git commit with the output.

## What is checked

- The committed corpus matches its frozen canonical SHA-256.
- The downloaded JSONL bytes match the export of that exact corpus before
  loading the Hub dataset.
- The loaded configuration, split, columns, row count, order, and every nested
  JSON value match the source, including raw responses and nullable provenance.
- The reconstructed corpus passes the ordinary corpus validator and retains
  its original canonical digest.
- Each loaded case runs through the ordinary verifier with explicit tolerance,
  sampling, timeout, and input-complexity settings. All reports, including
  inconclusive outcomes, are retained with package versions in the output.

There is **no silent normalization**: null padding, missing keys, number-type
changes, rewritten expressions, or altered labels fail the transport check.
The pinned Datasets version preserves the pilot's sparse domain and empty
parameter maps as JSON-valued features. The check does not rebuild fields from
the source to hide a changed loaded value.

The public data can also be loaded directly using Hugging Face's
[revision-pinned loading API](https://huggingface.co/docs/datasets/loading):

```python
from datasets import load_dataset

pilot = load_dataset(
    "oroikono/pdecert-pilot",
    name="default",
    split="test",
    revision="db690f9b161762ea288dd5dfb4b6b2f999c48e03",
    token=False,
)
```

## CI and evidence boundaries

`Hub pilot integration` runs offline installed-library tests on pushes and PRs.
Tests using mocked download/loader calls cover error paths only; they are not
evidence of a live Hub integration. Explicitly dispatching the workflow also
runs the anonymous live check. A Hub/network error fails that run instead of
being silently skipped. The normal core suite skips only the installed-library
test when optional dependencies are absent; the dedicated CI job installs them.

A successful command means the transport and verifier execution completed,
not that all candidates are valid. Inspect each report's status and evidence.
Exact symbolic results remain scoped to represented obligations; empirical
samples cannot prove a candidate. Stored annotations are never used to choose
the verifier's decision or modified by this check.

This is the frozen 20-row symbolic pilot, not a loader for Atlas v2, callables,
gridded fields, generated programs, or arbitrary Hub datasets. Its output is not
a new benchmark audit, historical timing reproduction, or a release attestation
binding the original generating code to a Hub commit. Independent labels and
that post-publication code/data receipt remain separate requirements.
