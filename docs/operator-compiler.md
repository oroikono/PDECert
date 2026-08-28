# Portable operator lowering

PDECert can lower the trusted operator expressions retained by a version 3
verification case into a PyTorch automatic-differentiation problem. This lets
one mathematical specification drive both an exact symbolic lane and an
empirical callable lane.

```python
from pdecert import compile_autodiff_problem, load_case

case = load_case("examples/exact_heat.json")
autodiff_problem = compile_autodiff_problem(case)
```

The candidate expressions stored in `case.fields` are not converted into a
neural model. A host supplies an independent `CallableCandidate`, such as a
trained PINN. The compiler translates only the trusted residual and condition
operators. The resulting callable report retains ordinary PDECert semantics:
sampled violations can be `REFUTED`, while sampled success remains
`INCONCLUSIVE`.

## Current contract

The first lowering accepts:

- parameter-free classical problems on rectangular domains;
- one or more named scalar fields;
- arithmetic, powers, coordinate and field references;
- expression derivatives through `D(expression, coordinate, order)`;
- consistent initial or boundary surfaces through
  `At(expression, coordinate, value)`;
- common PyTorch scalar functions including trigonometric, exponential,
  logarithmic, square-root, absolute-value, and error functions.

The lowering rejects before model evaluation:

- parameter variables, until callable parameter batching has explicit
  semantics;
- multiple incompatible `At` values for one coordinate in one obligation;
- a coordinate fixed by `At` but used outside that surface expression;
- symbolic functions without a faithful PyTorch lowering;
- constraints without retained, field-referenced operator sources.

Compilation never imports or executes candidate Python code. PyTorch is loaded
only when the compiled problem is evaluated, so symbolic-only installations do
not acquire a new dependency.

## Why this boundary matters

Previously, a contributor had to encode the heat equation twice: once as
schema expressions such as `D(u, t) - D(u, x, 2)`, and once as a handwritten
Python lambda over `AutodiffEvaluation`. Those two definitions could drift.
The compiler makes the schema expression the shared source while preserving
backend-specific evidence and reports.

This is a narrow compiler bridge, not a universal PDE language. UFL already
provides a mature variational-form DSL, and ModelingToolkit provides a common
symbolic PDE representation in the Julia SciML ecosystem. Future adapters
should translate documented subsets of established formats instead of claiming
that PDE representation itself is new. See
[`ADR-0006`](adr/0006-portable-operator-lowering.md) for the architectural
decision and unsupported semantics.

Run the end-to-end example with the optional backend:

```bash
pip install -e ".[dev,autodiff]"
python -m examples.compiled_heat
```
