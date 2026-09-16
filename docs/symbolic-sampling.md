# Symbolic sampling budgets

The symbolic verifier uses finite, deterministic coordinate samples to look
for violations of the represented PDE and conditions. This is an empirical
check, not a proof method or a guarantee of finding a defect.

## Sequence and compatibility

For each non-parameter coordinate on `(lower, upper)`, the first six normalized
locations remain `0.113, 0.271, 0.419, 0.613, 0.787, 0.937`. A smaller budget
uses that prefix, so the default five-point behavior is unchanged.

Additional locations are the midpoints of successive dyadic subdivisions:
`1/2`, then `1/4, 3/4`, then `1/8, 3/8, 5/8, 7/8`, and so on, stopping at the
requested count. Increasing the budget preserves the previous prefix. Each
fraction is mapped with `lower + (upper - lower) * fraction`.

This replaces the old behavior for budgets above six, which recycled the same
six fractions. Historical runs above that budget must be replayed using their
original code revision; their new results may differ. No frozen result is
rewritten by this change. Parameter sampling, callable/autodiff sampling, and
the separate fixed-collocation baseline retain their existing rules.

## Reproduce a missed residual

The deliberately constructed [example](../examples/sampling_coverage.json)
uses the classical problem `u_t = 0` and `u(x, 0) = 0` on the unit `(x, t)`
box. Its candidate is `u(x, t) = t * product(x - r)` over the six historical
fractions `r`. The initial condition holds, but the PDE residual is a nonzero
polynomial that vanishes at those six spatial locations.

From a checkout containing this fix:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_symbolic_sampling.py
pdecert verify examples/sampling_coverage.json --samples-per-axis 6 || test "$?" -eq 2
pdecert verify examples/sampling_coverage.json --samples-per-axis 7 || test "$?" -eq 1
```

With the default tolerance `1e-9`, six samples return `INCONCLUSIVE`: symbolic
checking does not establish a zero residual and no sampled violation exceeds
the tolerance. Seven samples add `x = 0.5` and return `REFUTED` with
`decision_evidence: EMPIRICAL` and the recorded residual witness. The example
is a regression fixture, not a natural model output or benchmark accuracy claim.

## Limits and contributor checks

- Passing any finite budget cannot discharge an obligation. Exact proof still
  requires a separate supported symbolic check.
- Floating-point rounding can map distinct fractions to identical coordinates
  or endpoints on very narrow or large-offset domains. The implementation
  generates the requested number of fractions, not a certified count of unique
  representable coordinates. It does not retry indefinitely to find unique
  floats. Overflow in extreme affine mappings is not addressed by this fix.
- Samples use a Cartesian product across variables. More points can be costly
  in several dimensions; no statistical coverage or convergence rate is claimed.
- Changes to this sampler should test prefix compatibility, ordinary-domain
  uniqueness, finite-precision limitations, exact proof, empirical refutation,
  and abstention. Preserve the provenance of earlier runs rather than changing
  their recorded configurations or outcomes to match a new sequence.
