# Atlas baseline adapters

PDECert baseline adapters reproduce familiar evaluation methods beside the
evidence-preserving verifier. Their outcomes are method-specific diagnostics,
not PDECert decisions. Both methods operate on the same validated Atlas v2
records and preserve the source digest and declared obligations.

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

## Fixed-collocation evidence contract

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

## Fixed-collocation scope

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

## Direct SymPy residual comparison

Run the second adapter on the same matched Atlas without PyTorch:

```bash
pdecert corpus baseline corpus/matched \
  --method direct-sympy \
  --symbolic-timeout 2 \
  --max-expression-ops 10000 \
  --output direct-sympy.json
```

The symbolic Fisher--KPP record produces four `zero` checks, one for the PDE and
three for its conditions. The callable record remains `unsupported`. The
command returns zero when it writes a report successfully, including reports
with `nonzero`, `undecided`, or unsupported records; inspect the results.

`DirectSympyBaseline` applies the existing pilot comparison's
`simplify` → `cancel` → `trigsimp` pipeline to every bound PDE residual and
represented initial/boundary condition. This uses established
[SymPy simplification](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html)
and its [three-valued assumptions queries](https://docs.sympy.org/latest/guides/assumptions.html).
It is a direct residual comparison, not an invocation of `checkpdesol` or a new
decision procedure.

Each check retains its stable obligation ID, original operator source,
materialized residual, simplified residual when available, and one outcome:

- `zero`: CAS establishes zero, recorded as `CAS_RESIDUAL_ZERO` / `EXACT`.
- `nonzero`: CAS establishes a finite real nonzero constant with no free
  symbols, recorded as `CAS_CONSTANT_NONZERO` / `EXACT`.
- `undecided`: no decision, with `ABSTENTION`, a reason, and no evidence level.

The record outcome is `nonzero` if any check has that outcome; otherwise it is
`zero` only if every check is zero, and `undecided` in all other cases. Checks
continue after a failure or timeout. Ordered IDs such as `pde_residuals[0]` and
`conditions[0]` retain each obligation's role and position. The existing problem
format still requires unique constraint names.

Exact evidence is confined to the represented expression under the template's
declared symbolic assumptions. This baseline performs **no domain singularity
or regularity checks**. Cancellation may erase a pole, and all-zero residuals
can coexist with an invalid candidate domain. There is no `PROVED` status,
solution-error bound, existence/uniqueness conclusion, or formal proof-kernel
certificate. General nonconstant residuals can remain undecided even when
another evaluator would find a counterexample.

### Scope and resource limits

The adapter accepts the classical symbolic lane with exact expression syntax
such as `1/2`. Any float literal in a candidate field or operator makes every
check undecided, including a float that could disappear during eager
simplification. Decimal domain bounds remain metadata and do not trigger that
rule. Nonfinite, non-real, or inexact residuals also abstain. Callable, weak,
entropy, gridded, and generated-program evaluation is outside this adapter.

Timeouts range from 0.001 to 3600 seconds; other values are rejected before
timer setup. One configured deadline covers template/candidate binding; a separate deadline
covers each residual's operation count, rendering, and simplification. An
unavailable deadline (including execution outside the main thread), an already
active real-time timer, an exception, or an operation-budget excess produces
undecided checks. A failed binding preserves one undecided entry per obligation.

The deadline is **not an end-to-end command or process limit**: Atlas loading
and validation happen first and already parse/bind symbolic candidates. The
operation budget bounds the admitted residual tree, not intermediate memory.
The signal-based timer does not provide process isolation, and total evaluation
time can grow with the number of obligations. Untrusted workloads still need
outer resource isolation.

### Report compatibility and Python extensions

Fixed collocation continues to emit report version 1 unchanged. Direct SymPy
opts into version 2, including when only unsupported records are selected.
The self-contained
[`version-2 schema`](../schema/atlas-baseline-report-v2.schema.json) accepts the
existing empirical rows and adds typed symbolic rows without mixing their
evidence vocabularies. Consumers should dispatch on `baseline_report_version`.

Existing adapters default to version 1 and keep returning `BaselineResult`.
An adapter returning `SymbolicBaselineResult` must declare `report_version = 2`
and return a complete ordered tuple of `SymbolicBaselineCheck` objects matching
the source Atlas's IDs, names, and operator expressions. The runner rejects
omitted, reordered, or rewritten obligations. Both built-in adapter IDs are
reserved for their respective versions, configuration, and accepted scope.
Serialize result objects through `to_dict()`.

Reproduction requires the unchanged Atlas, adapter ID/version, report version,
pipeline, exact-input/nonzero policies, symbolic timeout, operation budget, and
recorded Python/PDECert/SymPy versions. Hashes establish source identity only;
neither the report nor these contract checks establish independent labels.
