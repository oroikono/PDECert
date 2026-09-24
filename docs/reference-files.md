# Compare saved prediction and reference samples

Use `compare-reference` when you already have model predictions and reference
values at the same points. It runs the existing [reference-field metrics](reference-fields.md)
from a JSON file, without loading a model or importing an optional framework.
It is not an Atlas adapter or a gridded PDE checker.

## Try your own saved values

Install a development build containing `compare-reference` first; the older
PyPI release does not have this command. For example, from this checkout run
`python -m pip install .`. Then open an empty working directory. The following
block needs only the installed core package, not the checkout or any model
framework. It creates `samples.json`, compares the saved values, and prints one
field's metrics from `comparison.json`.

These are three made-up values to show the file format, not a trained model or
PDE benchmark. Replace the columns and metadata with your own aligned data.

<!-- tested-reference-walkthrough:start -->
```bash
python - <<'PY' &&
import json

samples = {
    "reference_input_version": 1,
    "coordinates": {"x": [0.0, 0.5, 1.0]},
    "candidate": {"u": [1.0, 2.0, 4.0]},
    "reference": {"u": [1.0, 2.0, 3.0]},
    "problem_id": "supplied-field-demo",
    "candidate_id": "deliberate-last-row-perturbation",
    "reference_id": "supplied-linear-values",
    "reference_description": "Values of 1 + 2*x at the supplied points",
    "reference_uncertainty": "Illustrative values, not a certified PDE solution",
    "sampling_description": "Three supplied points, equal row weights; no interpolation",
}
with open("samples.json", "x", encoding="utf-8") as destination:
    json.dump(samples, destination, indent=2, allow_nan=False)
PY
python -m pdecert.cli compare-reference samples.json --output comparison.json &&
python - <<'PY'
import json

with open("comparison.json", encoding="utf-8") as source:
    report = json.load(source)
print(json.dumps(report["fields"][0], indent=2, allow_nan=False))
PY
```
<!-- tested-reference-walkthrough:end -->

Each step runs only if the previous one succeeded. The input-creation step
refuses to overwrite an existing `samples.json`. Use a
fresh directory to repeat the whole block. The comparison command can replace an
existing `comparison.json` as described below. Keep `samples.json` with the report
so someone else can repeat the calculation.

The example returns RMSE `0.5773502691896257`, relative L2 approximately
`0.26726124`, and maximum error `1.0` at `x=1.0`. These are sample discrepancies,
not PDE counterexamples. Try these changes in the input dictionary:

| Change | Expected result |
| --- | --- |
| Set candidate values to `[1.0, 2.0, 3.0]` | Zero discrepancies, still empirical and not a proof |
| Set both value lists to `[0.0, 0.0, 0.0]` | RMSE zero; relative L2 is `null` with `zero_reference_norm` |
| Supply only two candidate values | Input error, exit `64`; no new report |
| Set `reference_input_version` to `2` | Unsupported version, exit `64`; no new report |

You can also compare the repository fixture from a checkout:

```bash
pdecert compare-reference examples/reference-input.json
pdecert compare-reference examples/reference-input.json --output comparison.json
```

You can use the same command from any working directory with an absolute input
path. The Python entry point is `pdecert.compare_reference_file(path)`. Both
paths produce the same report as `compare_reference_fields()` with the saved
columns and metadata.

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
empty. The command writes and syncs a temporary file in the destination directory,
closes it, then replaces the destination. Invalid input or a failed write, sync,
close, or replacement leaves an existing report unchanged. Temporary-file cleanup
is best effort; a cleanup failure is reported without hiding the original error.

Output must name a new file or an existing regular file, not a symlink, directory,
or special file. It must not name the input itself or a hardlink alias. The
destination directory must already exist and allow temporary-file creation and
replacement. On success, replacement creates a new file with temporary-file
permissions (owner-only on POSIX); previous permissions and other file metadata
are not preserved, and other hardlinks keep their old contents. These protections
assume a stable destination directory, not a concurrently modified or hostile
filesystem. They do not guarantee recovery after a power loss.

Keep the input file to replay a report. Its `input_sha256` still binds the
normalized binary64 columns and declared metadata, not the input file's raw
formatting or envelope version. It does not establish model execution,
checkpoint identity, reference truth, or independent labeling.
