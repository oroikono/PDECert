# Matched cases

A matched case binds two or more candidate artifacts to one declared
mathematical identity. It lets a benchmark compare symbolic expressions,
differentiable fields, and future gridded fields without pretending that their
verification backends provide equal guarantees.

Each `MatchedCase` declares:

- a stable case identifier;
- ordered coordinate and field names;
- the intended solution semantics, such as `classical`;
- two or more named `EvaluationLane` objects.

Each lane retains its native problem and artifact representation. PDECert checks
that the coordinate and field interfaces agree and that the problem/artifact
pair has a compatible verifier. The case author remains responsible for the
scientific assertion that the backend-specific problems encode the same PDE,
domain, parameters, and conditions. Constructing a matched case is not a proof
of equivalence between two problem specifications.

## Result semantics

`verify_matched_case` returns one report per lane and deliberately has no
aggregate status. For example, the same exact heat field can produce:

- `PROVED` in a symbolic lane because every supported obligation is an exact
  identity;
- `INCONCLUSIVE` in a callable lane because finite automatic-differentiation
  samples found no violation but cannot prove a continuous-domain claim.

Likewise, refuting one trained callable does not invalidate a different
symbolic candidate attached to the same problem identity.

Backend options are supplied per lane with `LaneVerificationOptions`. Symbolic
resource limits are rejected for callable lanes instead of being silently
ignored.

Run the complete symbolic/callable example with the optional PyTorch backend:

```bash
pip install -e ".[dev,autodiff]"
python -m examples.matched_heat
```

The example uses an analytical function represented both ways to demonstrate
the contract. Benchmark evidence about learned solvers requires independently
trained and frozen model artifacts; wrapping an analytical expression in a
PyTorch function is not a substitute.
