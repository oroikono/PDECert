# Pilot corpus

`pilot.json` contains 20 unedited generation records collected on 24 August
2026:

- 10 calls to SymPy 1.14.0 `pdsolve`, specialized from the returned
  characteristic function using the stated initial condition;
- 10 local generations from
  [`mlx-community/Qwen3-0.6B-4bit`](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit)
  at revision `73e3e38d981303bc594367cd910ea6eb48349da8`, using MLX LM 0.31.3.

All 20 annotations are intentionally `pending`. Automated verification is not a
substitute for the manual label and disagreement-resolution protocol.

## Collection procedure

Run the test suite first, then collect:

```bash
pip install -e ".[dev,collection]"
pytest
python -m experiments.collect_pilot
```

The default collection path runs the open model locally on Apple silicon. The
optional `--provider` path uses the base Qwen model through the named Hugging
Face inference provider and may consume paid or included credits.

The local protocol pins model and dependency revisions, disables thinking mode,
sets a deterministic seed for each attempt, and allows at most two repair turns
when a response is not a parseable explicit field. Every response from those
attempts is retained in the record's `raw_output` transcript. The final
expression is extracted with only these surface normalizations:

- remove a leading `u =` or `u(...) =` assignment;
- convert `^` to `**`;
- convert Unicode `π` and adjacent implicit products to `pi` and explicit `*`.

No mathematical term is added, removed, or corrected. The ten selected Qwen
transcripts are also stored individually under `raw/`. Their contents match the
corresponding record's `raw_output` exactly.

## Collection summary

- 20 records: 10 symbolic-solver and 10 open-model;
- 9 unique extracted expressions among the 10 open-model records;
- 8 open-model records completed in one attempt and 2 used one repair turn;
- all raw-output SHA-256 digests validate;
- machine verification returns `PROVED` for the 10 specialized solver outputs
  and `REFUTED` for the 10 open-model outputs.

These machine results describe the current verifier, not the corpus labels.

## Licensing and provenance

PDECert's original metadata and completed annotations are distributed under the
repository's MIT license. Every record separately names the source producer,
source URL, version or revision, and the license recorded for that source
software or model. A source license does not by itself settle every right in a
generated output, so downstream users must also inspect the applicable source
and provider terms. The eventual Hub dataset card uses a mixed-provenance
license marker instead of presenting every component as if it shared one
license.
