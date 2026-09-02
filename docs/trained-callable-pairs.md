# Trained callable pairs

PDECert includes two matched nonlinear cases. Each case binds one symbolic
candidate and one separately trained physics-informed neural network to the same
candidate-free problem template. The purpose is to compare evidence, not to
rank solvers or transfer correctness from one artifact to another.

## Cases

### Viscous Burgers traveling wave

The first pair uses

\[
u_t + u u_x - 0.1 u_{xx} = 0,
\qquad (x,t) \in [-1,1] \times [0,1].
\]

Its symbolic field is a declared exact classical solution. Its callable field
is a separately initialized dense tanh network trained from the PDE residual
and the declared initial and boundary traces. The committed evaluation exactly
proves the symbolic lane and empirically refutes the PINN with a PDE-residual
witness of approximately `2.34e-2` at the configured tolerance.

### Fisher--KPP traveling front

The second pair uses

\[
u_t - u_{xx} - u(1-u) = 0,
\qquad (x,t) \in [-6,6] \times [0,2].
\]

The symbolic candidate is the byte-preserved output of
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The prompt contains the expected
traveling-front form in its boundary traces, so this case does not measure
unguided PDE discovery. Its Atlas annotation remains `pending`: PDECert's exact
report is machine evidence, not an independent human label.

The paired 2-by-16 dense tanh PINN was trained for 3,000 deterministic CPU
float64 Adam steps using only PDE, initial, and boundary losses. It did not use
interior exact field values. Its recorded final losses are:

| Loss | Value |
| --- | ---: |
| PDE mean squared residual | `1.6717e-5` |
| Initial-condition MSE | `5.5698e-7` |
| Combined boundary MSE | `5.4854e-7` |
| Weighted total | `2.7772e-5` |

At the predeclared evaluation tolerance `1e-3`, the symbolic candidate is
`PROVED` with `EXACT` evidence. The PINN is `REFUTED` with an `EMPIRICAL`
automatic-differentiation witness: PDE residual `9.1209e-3` at
`x=-2.748`, `t=0.226`. This contrast is the result: low collocation loss is not
a continuous-domain certificate and does not prevent a held-out refutation.

Both training scripts follow the physics-informed collocation pattern of
[Raissi, Perdikaris, and Karniadakis](https://arxiv.org/abs/1711.10561). The
fixtures are evaluation targets, not new solver methods.

## Evidence contract

- Scope: declared classical strong-form obligations on each rectangular domain.
- Symbolic lane: exact identities may produce `PROVED` for that expression.
- Trained callable lane: deterministic automatic-differentiation samples may
  produce a replayable empirical refutation. Passing samples remains
  `INCONCLUSIVE`.
- Excluded: weak, entropy, or stability claims; irregular geometries; uniqueness;
  global neural-network guarantees; and any inference that residual size bounds
  solution error.

Matched reports deliberately have no combined status. Evidence from an exact
lane is never transferred to a trained model.

## Portable artifact boundary

Frozen callables are plain JSON, not pickles or executable checkpoints. Version
1 supports only a bounded CPU float64 dense tanh MLP with one output. It records
coordinate and field names, full weights, training configuration, final losses,
software version, training-script digest, and canonical weight digest. The
loader rejects unknown fields, unsupported architecture choices, ragged or
incorrect tensor shapes, non-finite values, duplicate JSON keys, and digest
mismatches before importing PyTorch.

A companion integrity record binds artifact bytes, configuration, weights, and
repository-contained sources. Portable integrity version 2 identifies an
executor and run without assuming one cluster; the historical Burgers version-1
record remains readable. These digests establish byte identity only. They do
not establish authorship, trusted execution, scientific correctness, or that a
training script produced the declared weights. See
[`frozen-callables.md`](frozen-callables.md) for the representation contract.
The Fisher--KPP runner rejects active template, raw-output, case, or record paths
that are not the digest-bound inputs, binds the decision-relevant evaluator
sources, and records Python, platform, PDECert, SymPy, and PyTorch versions in
the result.

## Reproduce

Install the optional backend and evaluate both committed pairs:

```bash
pip install -e ".[dev,autodiff]"
python -m experiments.burgers_pinn_fixture \
  benchmarks/matched/burgers-classical-01/pinn.json \
  benchmarks/matched/burgers-classical-01/integrity.json
python -m experiments.trained_burgers_pair
python -m experiments.trained_fisher_kpp_pair
```

To reproduce Fisher--KPP training without replacing frozen evidence:

```bash
python -m experiments.train_fisher_kpp_pinn \
  --output /tmp/fisher-kpp-pinn-reproduction.json
```

CPU training is deterministic for the recorded software stack, but exact
weights can still change across PyTorch versions or platforms. Reproduction
should compare configuration, losses, field behavior, and verifier outcomes;
it should not assume a bitwise-identical optimization trajectory. A new artifact
does not replace a committed reference until its integrity record is reviewed.
