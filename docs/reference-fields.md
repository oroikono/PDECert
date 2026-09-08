# Comparing supplied reference fields

`compare_reference_fields()` is a dependency-free metric primitive for already
evaluated candidate and reference values at the same supplied points. It helps
integrations report prediction errors without converting a native model to
PDECert's frozen-MLP representation. It does not execute a model, check a PDE,
validate a problem template, or decide whether either field is a solution.

The report is explicitly `EMPIRICAL`. It has no `PROVED`, `REFUTED`, pass/fail,
tolerance decision, independent label, or true-solution error bound. This is a
primitive for future reference-field Atlas adapters, not a completed adapter or
an implementation of the gridded-artifact lane.

## Minimal use

```python
from pdecert import compare_reference_fields

report = compare_reference_fields(
    coordinates={"x": [0.0, 0.5, 1.0]},
    candidate={"u": [1.1, 2.1, 3.1]},
    reference={"u": [1.0, 2.0, 3.0]},
    problem_id="my-problem-v1",
    candidate_id="my-model-samples-v1",
    reference_id="my-reference-v1",
    reference_description="Values from a separately retained reference calculation",
    reference_uncertainty="Reference solver and discretization errors are not bounded",
    sampling_description="Three supplied points, equal weight per row; no refinement study",
)
print(report["fields"])
```

From a repository checkout, run the complete deterministic example:

```bash
python -m examples.reference_field_comparison
```

It compares a Fisher--KPP traveling-front formula with a deliberate `0.02*t`
perturbation at 25 interior points. Expected RMSE is approximately `0.02370654`,
relative L2 is `0.04141737`, and sampled maximum error is `0.038`. The formula
is evaluated in binary64; this command does not certify it against the PDE.
This is an injected analytical fixture, not a trained-model failure, external
reproduction, or independently labeled benchmark result.

For actual trained-model outputs, the optional
[Fisher--KPP walkthrough](trained-reference-comparison.md) evaluates the existing
frozen PINN and places reference errors beside fresh PDE and trace diagnostics.
It retains every sample and checks every obligation even after an interior
failure. It neither retrains the model nor upgrades empirical evidence.

## Inputs and native workflows

Each mapping contains named, flat Python numeric sequences, such as lists or
tuples. Every coordinate, candidate field, and reference field must contain the
same positive number of values. Candidate and reference field names must match
exactly. No broadcasting, truncation, interpolation, or implicit flattening is
performed. Multiple named fields are compared separately.

For a NumPy prediction matrix, explicitly supply a field as
`prediction[:, j].tolist()`. For PyTorch use
`prediction[:, j].detach().cpu().tolist()`. These are conversion recipes, not
tested model loaders or a DeepXDE/PyTorch integration. Retain model inference
settings, checkpoint identity, dtype/device, coordinate ordering, and any input
or output transforms with the sample files. The metric API cannot recover them.

The caller must align every row, undo normalization when needed, use compatible
physical coordinates and units, and justify reference quality. `problem_id` is
a declared identifier, not a validated link to a mathematical specification.
All six metadata strings are required and must be nonblank. In particular,
reference uncertainty cannot be silently omitted. The report marks its reference
assessment as `caller_supplied_not_verified`.

## Metric definitions

For one field with `n` rows, candidate values `c_i`, reference values `r_i`, and
differences `e_i = c_i - r_i`, the metrics are:

- RMSE: `sqrt(sum(e_i**2) / n)`;
- discrete relative L2: `sqrt(sum(e_i**2)) / sqrt(sum(r_i**2))`;
- sampled maximum absolute error: `max(abs(e_i))`.

These are established prediction metrics, not new certification methods. See
[DeepXDE's relative-L2 metric](https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.metrics.l2_relative_error)
for an existing SciML interface. PDECert implements its reductions independently
with scaled arithmetic and explicit unavailable-value reasons.

Rows have equal weight; duplicate coordinates retain their multiplicity. These
are discrete sample metrics, not volume-weighted quadrature or continuous-domain
norms. Fields are not aggregated because their units and scales may differ.
`max_error_sample` records the first row attaining the computed maximum, its
coordinates, and both values. It is a discrepancy location, not a PDE refutation
witness. Even zero discrepancy means only agreement with this reference at
these points.

## Numerical and resource limits

The implementation uses Python binary64 and scaled sums of squares. It rejects
booleans, complex or nonfinite values, strings, nested arrays, generators, and
integers that lose precision on conversion to binary64. Already rounded floats
cannot be restored or assessed for earlier loss of precision. Convert framework
arrays explicitly; they are not implicitly loaded or executed.

A zero reference norm makes relative L2 unavailable, including when both vectors
are zero. The report uses JSON `null` and `zero_reference_norm`. A relative metric
that exceeds the binary64 range becomes `null` with `binary64_overflow`; a nonzero
RMSE or relative value that rounds to zero becomes `null` with
`binary64_underflow`. An unrepresentable absolute difference rejects the entire
comparison with `ReferenceComparisonError`; it does not return a partial report.
No NaN or infinity is emitted. Finite values still have ordinary rounding error
and are not rigorous enclosures.

At most `REFERENCE_COMPARISON_MAX_VALUES` (1,000,000) total scalar values across
coordinates, candidate, and reference columns are accepted. Shapes and budget
are checked before numeric conversion. This is a workload guard, not a memory
sandbox or isolation from untrusted Python objects. Metadata byte sizes and
custom object behavior are not sandboxed.

## Report identity and reproduction

`input_sha256` binds the normalized binary64 samples and declared metadata using
sorted JSON keys and compact separators. Column names are sorted, signed zero
is normalized to positive zero, and row order is preserved. Exactly representable
integers and their float equivalents share identity. Retain all original sample
columns and metadata to replay the calculation; the report does not embed the
full sample table.

The digest does not bind checkpoint bytes, source execution, or truth of the
declared reference. The report also records package/Python versions, arithmetic,
reduction, units policy, and value budget. Retain the exact code revision for
unreleased development versions and source-framework versions for model samples.

[`schema/reference-comparison-v1.schema.json`](../schema/reference-comparison-v1.schema.json)
validates report structure and local metric/reason consistency. It does not
recompute metrics, establish reference quality, or check all cross-field
relations in an edited report. Existing verifier reports and Atlas baseline
formats are unchanged.
