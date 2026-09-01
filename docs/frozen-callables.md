# Frozen callable artifacts

PDECert's frozen-callable format carries a trained model as restricted JSON. It
lets a contributor preserve model identity and later materialize a differentiable
candidate without accepting a pickle, a Python module, or artifact-supplied code.
Loading, hashing, writing, and integrity validation do not import PyTorch.
The validated record implements PDECert's public solution-artifact identity
protocol; materialization is a separate, explicit step.

## Version 1 scope

Version 1 accepts only a dense MLP with one to four hidden layers, `tanh` hidden
activations, CPU execution, and `float64` parameters. A layer width is between 1
and 128. The architecture declares one to eight unique identifier input names in
column order and exactly one identifier output name. State-dictionary keys and
tensor shapes are derived from that declaration and must match exactly.
Artifact and integrity JSON files are limited to 4,000,000 bytes, 32 nested
levels, and 200,000 JSON values. Integrity records bind at most 64 source files.
Decoder recursion and resource-limit failures are reported as
`FrozenCallableError` rather than escaping as implementation exceptions.

The artifact uses the same top-level and training fields as the original frozen
Burgers PINN fixture. Training metadata and weights must be JSON values with
finite numbers. Unknown or missing fields, ragged arrays, non-finite values,
unsupported architectures, and digest mismatches are rejected. The normative
shape checks are implemented by the Python loader; the JSON Schema provides a
portable structural precheck.

```python
from pdecert import (
    load_frozen_callable,
    materialize_frozen_callable,
    validate_frozen_callable_integrity,
)

artifact = load_frozen_callable(
    "benchmarks/matched/burgers-classical-01/pinn.json"
)
validate_frozen_callable_integrity(
    "benchmarks/matched/burgers-classical-01/pinn.json",
    "benchmarks/matched/burgers-classical-01/integrity.json",
)
candidate = materialize_frozen_callable(artifact)
```

Materialization constructs the declared network with PDECert's own code, loads
only validated numeric tensors, selects evaluation mode, and disables parameter
gradients. The JSON never selects a Python name or supplies executable source.
PyTorch is optional until this step; install PDECert's `autodiff` extra to
materialize and evaluate a candidate.

## Integrity and evidence meaning

`weights_sha256` identifies canonical JSON tensor values. The companion integrity
record binds the artifact's exact bytes, declared training configuration, and
repository-contained source files. Portable integrity version 2 records an
executor and run identifier without assuming a particular cluster. The
historical Burgers version-1 record with its Euler job identifier remains
readable. In both versions, the declared training script must be among the
source files and match its recorded digest. Absolute paths, repository
traversal, and symlink escapes are rejected.

These hashes establish identity only. They do not establish who ran training,
that the metadata is true, that the source produced the weights, or that the model
satisfies a PDE. Independent provenance controls are still required for those
claims.

A materialized candidate remains an empirical callable. A valid sampled
violation may refute the candidate against the sampled obligation. Passing
samples, small residuals, or successful loading remain `INCONCLUSIVE` for a
continuous-domain solution claim unless a separate rigorous backend and its
assumptions establish stronger evidence.
