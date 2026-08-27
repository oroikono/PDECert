# Trained callable pairs

PDECert's first trained callable fixture is a small physics-informed neural
network paired with an exact symbolic traveling wave for the viscous Burgers
equation

\[
u_t + u u_x - 0.1 u_{xx} = 0,
\qquad (x,t) \in [-1,1] \times [0,1].
\]

The symbolic field is an exact classical solution. The callable field is a
separately initialized dense tanh network trained from the PDE residual and the
declared initial and boundary traces. It is not an analytical function wrapped
in a PyTorch callable. This follows the physics-informed collocation pattern of
[Raissi, Perdikaris, and Karniadakis](https://arxiv.org/abs/1711.10561), but the
fixture is an evaluation target, not a new solver method.

## Evidence contract

- Scope: classical strong-form obligations on the rectangular domain above.
- Symbolic lane: exact identities may produce `PROVED`.
- Trained callable lane: deterministic automatic-differentiation samples may
  produce a concrete empirical refutation. Passing samples remains
  `INCONCLUSIVE`.
- Excluded: weak or entropy solutions, inviscid discontinuities, irregular
  geometries, and any claim that a residual value bounds the solution error.

The matched report deliberately has no combined status. Evidence from the
exact lane is not transferred to the trained model.

## Artifact boundary

The committed PINN is plain JSON rather than a pickle or executable checkpoint.
It records the architecture, coordinate and field names, full float64 weights,
training configuration, final training losses, PyTorch version, training-script
digest, and weight digest. A companion integrity record binds the exact artifact
bytes, configuration, weights, training script, restricted loader, evaluator,
Euler job, and base revision. The loader accepts only a bounded dense tanh
architecture and validates every tensor shape and finite value before creating
an evaluation-only module. Unsupported activations, layouts, devices, and
malformed weights are rejected.

## Reproduce

Install the optional backend and validate the committed evidence before running
the matched evaluation:

```bash
pip install -e ".[dev,autodiff]"
python -m experiments.burgers_pinn_fixture \
  benchmarks/matched/burgers-classical-01/pinn.json \
  benchmarks/matched/burgers-classical-01/integrity.json
python -m experiments.trained_burgers_pair
```

To reproduce training without replacing the frozen evidence, write to a new
path:

```bash
python -m experiments.train_burgers_pinn \
  --output /tmp/burgers-pinn-reproduction.json
```

CPU training is deterministic for the recorded software stack, but exact
weights may still change across PyTorch versions or platforms. Reproduction
should compare configuration, losses, field behavior, and verifier outcomes;
it should not assume bitwise-identical optimization trajectories. A newly
trained artifact is intentionally not accepted as the committed reference until
a new integrity record is reviewed.
