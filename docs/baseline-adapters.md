# Atlas baseline adapters

PDECert baseline adapters reproduce familiar evaluation methods beside the
evidence-preserving verifier. Their outcomes are method-specific diagnostics,
not PDECert decisions.

The first adapter is deterministic full-condition fixed collocation for Atlas
v2 symbolic expressions:

```bash
pdecert corpus baseline corpus/matched \
  --method fixed-collocation \
  --points-per-axis 5 \
  --decimal-precision 30 \
  --tolerance 1e-9 \
  --output fixed-collocation.json
```

The command validates the complete Atlas before evaluating any selected record.
It binds each symbolic artifact to its candidate-free template, evaluates every
represented PDE residual and initial or boundary condition on a uniform tensor
grid including domain endpoints, and records the largest absolute residual.
Use `--record` repeatedly to select a subset.

## Outcome and evidence contract

The report deliberately does not contain a PDECert `status`, an aggregate
verdict, accuracy, or truth labels.

- `pass` means that no grid value exceeded the configured tolerance. It carries
  `EMPIRICAL_PASS` evidence and does not establish the obligation between grid
  points.
- `fail` carries a `NUMERICAL_THRESHOLD_EXCEEDANCE` with the original
  constraint source, values for the residual's free variables, and absolute
  residual at the worst sampled input. For conditions, the retained `At(...)`
  source records the actual initial or boundary surface.
- `unsupported` carries an `ABSTENTION` reason and no residual or witness.

In particular, `pass` never becomes `PROVED`. A hidden singularity, narrow
boundary defect, high-frequency alias, or localized residual can evade a fixed
grid. A numerical failure is reproducible evidence that this implementation
exceeded its threshold, not automatically a mathematical refutation or a
solution-error estimate. Roundoff can cause false failures when tolerance is
too small for the configured decimal precision; important failures should be
checked for precision stability or independently reproduced.

### Arithmetic precision and replay

The grid and its saved witness coordinates use Python binary64 floats. Before
evaluating a residual, float coordinates are converted directly to mpmath
numbers inside the configured `decimal_precision` context. Integer parameter
coordinates remain exact Python integers. This prevents polynomial arithmetic
from silently running at binary64 precision even when more digits were requested.
The final absolute residual is converted back to a JSON-compatible float (or
`"infinity"` when nonfinite or outside that range).

Conversion uses the actual binary coordinate, not its printed decimal spelling.
Increasing arithmetic precision does not refine the grid or recover lost input
digits. To replay a saved witness, convert float `sampled_inputs` with
`mpmath.mpf(value)` inside `mpmath.workdps(decimal_precision)` before calling the
mpmath-generated residual; leave integer inputs unchanged.

Run the manufactured cancellation example:

```bash
python -m examples.collocation_precision
```

Its transport residual expands to `x**2 - 200000000*x + 10000000000000001`,
which is `(x - 100000000)**2 + 1`. At the five sampled coordinates in
`[100000000, 100000001]` its exact values are `1, 1.0625, 1.25, 1.5625, 2`.
At 30 digits the adapter reports `fail`, maximum residual `2`, and endpoint
witness `x = 100000001`. Native float evaluation previously rounded every one
of these values to zero despite the requested precision. No initial or boundary
conditions are included: this example isolates evaluation arithmetic, not
problem coverage or a natural solver failure.

This corrects the existing precision contract without changing adapter or
report versions. Cancellation-sensitive results from older builds should be
rerun with the corrected implementation; preserve the software revision when
comparing them. Finite precision remains empirical: low precision can still
miss defects or create threshold exceedances, and no fixed digit count is a
universal accuracy guarantee. The mpmath context is process-global, so concurrent
threaded runs with different precisions are not isolated; use separate processes.

This fix does not change the handling of integer-only expressions. Although
integer inputs themselves remain exact, division or negative powers in the
generated function can still create native-float intermediates. For example,
`1/n - 1/3` at integer `n = 3` can show a small roundoff residual even when more
digits are requested. Such exceedances need independent or precision-aware
replay; changing `decimal_precision` alone need not resolve every arithmetic
path.

## Accepted and unsupported scope

Version 1 accepts Atlas v2 `symbolic_expression` records using
`classical_strong` templates on finite rectangular domains. It includes every
represented condition and supports the template parameter assumptions already
accepted by PDECert; integer parameters receive a deterministic integer grid.

Callable models, gridded fields, generated programs, weak or entropy semantics,
nonrectangular domains, validated continuous-domain bounds, and solution-error
guarantees are unsupported. A callable record in a mixed Atlas is retained as
an `unsupported` row rather than silently omitted.

## Reproduction and extension boundary

Each report records the canonical Atlas SHA-256, adapter ID and version, grid
density, tolerance, decimal precision, one-million-evaluation resource limit,
sampling rule, condition policy, and Python, PDECert,
SymPy, and mpmath versions. The public JSON contract is
[`schema/atlas-baseline-report-v1.schema.json`](../schema/atlas-baseline-report-v1.schema.json).
The Atlas digest establishes content identity only.

Python integrations implement the explicit `AtlasBaselineAdapter` protocol and
return validated `BaselineResult` objects. New methods must define their own
accepted artifacts and solution semantics, reproduction settings, evidence
strength, replayable failure witness, and abstention boundary. Adding an
adapter does not change the built-in verifier or make the method a proof
backend.

The runner checks each record against the adapter's declared artifact types and
solution semantics before calling it. Out-of-scope records receive an
`unsupported` row even if the adapter does not implement its own scope guard.
Supported records are passed as private copies; an adapter that changes the
supplied record is rejected. Keep any normalization or intermediate state in
separate objects. The report digest and record identifiers always refer to the
original, validated Atlas.
The `fixed_collocation` adapter ID is reserved for its versioned configuration
and scope; external methods must choose their own ID.

`BaselineResult` rejects infinite passing residuals, non-null reasons on pass or
fail outcomes, and failures without a typed witness whose residual matches the
reported maximum. `BaselineWitness` copies and freezes its sampled inputs so
later changes to an adapter's working dictionary cannot alter saved evidence.
These checks enforce the report contract; Python adapters are trusted code and
are not sandboxed or independently verified by the runner.
