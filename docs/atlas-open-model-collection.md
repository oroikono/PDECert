# Open-model Atlas collection

Natural-candidate batches are declared before inference. A batch manifest fixes
the model revision, generation settings, PDE cases, domains, conditions, and
expected field names. This prevents choosing only outputs that make one checker
look favorable.

The first batch is
`experiments/atlas_batches/qwen3_1_7b_batch01.json`. It contains eight cases
covering elliptic, parabolic, hyperbolic, nonlinear, dispersive, parameterized,
two-dimensional, and coupled problems.

## Environment

Install the collection environment separately from the core library:

```bash
python -m venv .venv-collection
.venv-collection/bin/pip install -r requirements/atlas-collection.txt
.venv-collection/bin/pip install -e .
```

Model weights and run transcripts can be large. Keep them outside the Git
checkout, such as in a cluster scratch directory.

## Generate once

```bash
python -m experiments.collect_atlas_batch generate \
  experiments/atlas_batches/qwen3_1_7b_batch01.json \
  --run-directory /path/to/scratch/qwen3-1.7b-batch-01
```

Generation is greedy and the manifest pins a full model revision. Each case is
written immediately. Existing transcripts are skipped, making an interrupted
job resumable without silently rerunning earlier cases.

Do not edit model responses. Do not rerun a case because its answer is
mathematically inconvenient. If a response cannot be parsed, retain the
transcript and publish the `not_materialized` outcome in the collection report.

## Materialize records

```bash
python -m experiments.collect_atlas_batch materialize \
  experiments/atlas_batches/qwen3_1_7b_batch01.json \
  --run-directory /path/to/scratch/qwen3-1.7b-batch-01 \
  --atlas corpus/community \
  --report results/qwen3-1.7b-batch-01-collection.json

pdecert corpus validate corpus/community
```

The materializer accepts only the declared model, revision, settings, prompt,
case identifier, and fields. It preserves the decoded response byte-for-byte as
`raw-output.txt`. Surface extraction may remove a leading assignment and
normalize `^` or Unicode pi; it does not repair mathematics.

Every materialized record remains `pending`. Run human review under
`corpus/LABELING.md` before exposing verifier or baseline outcomes to the
reviewer. Collection success, parseability, and mathematical validity are
separate questions.
