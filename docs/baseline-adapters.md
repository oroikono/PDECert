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
