# Compare saved prediction and reference samples

Use `compare-reference` when you already have model predictions and reference
values at the same points. It runs the existing [reference-field metrics](reference-fields.md)
from a JSON file, without loading a model or importing an optional framework.
It is not an Atlas adapter or a gridded PDE checker.

From a checkout of this development version:

```bash
python -m pip install -e .
pdecert compare-reference examples/reference-input.json
pdecert compare-reference examples/reference-input.json --output comparison.json
```

The example deliberately changes the last of three values. It returns RMSE
`0.5773502691896257`, relative L2 approximately `0.26726124`, and maximum error
`1.0` at `x=1.0`. These are sample discrepancies, not PDE counterexamples.

You can use the same command from any working directory with an absolute input
path. The Python entry point is `pdecert.compare_reference_file(path)`. Both
paths produce the same report as `compare_reference_fields()` with the saved
columns and metadata. The new command is not part of the older PyPI release.

## Input contract

Start from [`examples/reference-input.json`](../examples/reference-input.json).
The version-1 envelope requires these keys and rejects any others:

- `reference_input_version`: the integer `1`;
- `coordinates`, `candidate`, `reference`: nonempty objects of named numeric
  lists, with equal lengths and matching candidate/reference field names;
- `problem_id`, `candidate_id`, `reference_id`, `reference_description`,
  `reference_uncertainty`, `sampling_description`: nonblank strings.

The caller must align the rows, use compatible units, and disclose reference
quality and sample selection. There is no broadcasting, flattening,
interpolation, checkpoint loading, or reference-quality assessment. Retain
the original model, inference settings, transforms, and source data separately.

The reader accepts regular UTF-8 files, rejects duplicate object keys and
nonfinite numbers, and applies the Python API's binary64 and one-million-scalar
limits. It reads at most `REFERENCE_INPUT_MAX_BYTES + 1` bytes and rejects files
over 64 MiB before parsing. Invalid UTF-8, lone surrogate characters, malformed
or excessively nested JSON, missing metadata, and unsupported versions fail
with `ReferenceComparisonError`. File-access failures can raise `OSError`.
These guards are not a process, memory, timeout, or concurrent-filesystem sandbox.

The [input schema](../schema/reference-input-v1.schema.json) describes structure;
the reader additionally checks cross-column lengths, binary64 representability,
file size, and aggregate value limits. JSON Schema treats `1.0` as an integer;
the reader deliberately requires the integer token `1`. Use the reader as well
as a schema check before relying on an input file.

## Results and exit codes

The report retains `EMPIRICAL` evidence and has no PDE status, tolerance
decision, or true-solution error bound. Even zero error only means agreement at
the supplied points. A zero reference norm leaves relative L2 unavailable with
a recorded reason; it is not a file error.

Exit `0` means the comparison completed, including when values disagree or a
metric is unavailable. It never means `PROVED`. Invalid input and file-write
errors return `64` with a message on stderr. Argument-usage errors return `2`.
With `--output`, successful execution writes JSON to that file and leaves stdout
empty. Invalid input does not replace an existing report. Output may replace an
existing report, but must not name the input itself or a symlink/hardlink alias.

Keep the input file to replay a report. Its `input_sha256` still binds the
normalized binary64 columns and declared metadata, not the input file's raw
formatting or envelope version. It does not establish model execution,
checkpoint identity, reference truth, or independent labeling.
