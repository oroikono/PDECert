# Audit your own differentiable model

`audit_callable()` checks an already-loaded PyTorch model against a trusted
candidate-free `ProblemTemplate`. It runs every represented PDE, initial, and
boundary obligation separately and returns one JSON-compatible report. A
sampled violation can refute an obligation; a sampled pass remains
`INCONCLUSIVE`. This is an empirical diagnostic, not neural-network
certification or a bound on solution error.

## Install and try the trained example

Run from a checkout containing this API; the published `pdecert==0.1.1rc2`
release does not include `audit_callable()`:

```bash
python -m pip install -e ".[dev,autodiff]"
python -m examples.audit_trained_model > model-audit.json
```

The example evaluates the bundled, previously trained Fisher--KPP model stored
in restricted frozen JSON. It does not train a model, download weights, or call
a model provider. Its report describes that fixture, not your model. See the
[trained-pair guide](trained-callable-pairs.md) for its retained provenance and
limitations.

## Wrap your existing model

First define the mathematical task in a
[problem template](problem-templates.md): its coordinate order, domains,
named fields, PDE residuals, and initial or boundary conditions. The model must
not choose or modify these obligations. Use the template for the problem your
model actually represents.

The following adaptation assumes `model` is your already-loaded scalar heat
model, on CPU with float64 parameters, and that its native input and output
already use the template's coordinates and units. The bundled heat template
orders input columns as `(x, t)` and declares the field `u`. Replace it with
your own template and wrapper as appropriate; this snippet does not load a
checkpoint.

```python
import json

from pdecert import CallableCandidate, audit_callable, load_template

template = load_template("examples/heat-template.json")
model.eval()


def u(points):
    return model(points)  # shape (number_of_points, 1)


artifact = CallableCandidate.from_mapping(
    {"u": u},
    dtype="float64",
    device="cpu",
)
provenance = {
    "coordinate_mapping": "identity; columns x,t and output u use template units",
}
report = audit_callable(
    template,
    artifact,
    candidate_id="my-heat-model",
    provenance=provenance,
    tolerance=1e-6,
    samples_per_axis=5,
    max_sample_points=100_000,
)
print(json.dumps(report, indent=2, allow_nan=False))
```

Each field receives a tensor of shape `(points, variables)` in
`template.variables` order and must return `(points,)` or `(points, 1)`.
Supply exactly the template's field names: missing and extra fields are errors.
For a model with several output columns, map each declared scalar field to its
corresponding column without detaching the result.

The artifact requires an explicit `dtype` (`"float32"` or `"float64"`) and
`device`. They describe the tensors used for evaluation; they do not move or
convert your model. Choose settings compatible with its parameters and native
operations, and retain those choices with the report.

If training used normalized coordinates, put the physical-to-model transform
inside the wrapper using differentiable tensor operations. Likewise, undo any
output scaling there so the returned value is the physical field used in the
template. Automatic differentiation must see these transforms to apply the
chain rule. PDECert cannot infer units or detect an omitted transform. Avoid
`predict()` paths that return NumPy arrays, detach tensors, or disable gradients.

## Callable assumptions

The supported lane uses classical strong, pointwise PyTorch derivatives over
finite rectangular domains. The callable compiler rejects parameters and
operators outside its supported subset; see the
[operator compiler](operator-compiler.md). Other model backends, gridded output,
weak solutions, and irregular geometry need their own supported representation.

Each output row must depend only on the corresponding input row. Use a
deterministic evaluation model that is differentiable to the orders requested
by the template. Cross-sample attention, batch-dependent behavior, randomness,
and mutable state can invalidate this interpretation. `model.eval()` sets
module evaluation behavior; gradients must still be enabled. Run the audit
outside `torch.no_grad()` and `torch.inference_mode()`.

Directly supplied `torch.nn.Module` fields are rejected if the module or any
child is in training mode. The helper cannot inspect a model hidden inside a
Python closure, such as `u` above, or establish that arbitrary callable code is
pointwise and correctly differentiable. Those remain the caller's responsibility.
Model execution is trusted Python code, not a sandboxed checkpoint-loading path.

## Read every obligation

The report has `audit_version: 1` and no overall status. It retains the
template, candidate ID and field names, caller-supplied provenance, evaluation
settings, environment, and an `obligations` list. Each obligation records its
source ID, kind, name, expression, fixed coordinates, actual sample points,
and a native `diagnostic_report`.

```python
for obligation in report["obligations"]:
    diagnostic = obligation["diagnostic_report"]
    print(
        obligation["source_obligation_id"],
        obligation["name"],
        diagnostic["status"],
    )
```

Each native report addresses a fresh single-obligation problem, so its local
ID is `constraint:0`. Use the enclosing `source_obligation_id` to map evidence
back to the original template. An interior failure does not suppress the
initial or boundary checks. Keep the full reports, including witnesses and
incomplete reasons, rather than reducing them to a single verdict.

`samples_per_axis` controls deterministic finite samples along each unfixed
coordinate; surface obligations fix their declared initial or boundary
coordinate. `max_sample_points` bounds the total rows across all obligations
before model evaluation. It is not a wall-time or memory limit. The recorded
grids are also checked before any model call: nonfinite or out-of-domain
dtype-cast points are rejected, including interior points rounded onto a boundary.
Choose a compatible precision rather than silently moving those samples. The
points support replay, but finite coverage can miss defects between points.
Tolerance and floating-point arithmetic affect threshold decisions: a
`REFUTED` report is an empirical witness under those settings, and passing
samples remain `INCONCLUSIVE` regardless of sample count.

## Retain truthful provenance

The helper records your string-to-string `provenance` mapping. Include actual
checkpoint, training-run, model revision, and preprocessing information when
available; do not invent unavailable metadata. For example, compute a digest
of the actual checkpoint file before calling the helper:

```python
import hashlib
from pathlib import Path

checkpoint_path = Path("path/to/your/actual-checkpoint")
provenance["checkpoint_sha256"] = hashlib.sha256(
    checkpoint_path.read_bytes()
).hexdigest()
```

This is a claim about those file bytes. The helper does not verify that the
in-memory model was loaded from that checkpoint, and a digest establishes
neither authorship nor mathematical correctness. Preserve the template,
wrapper and transforms, original checkpoint, full report, and environment
needed to replay the calculation. Training loss and sampled passes do not
provide independent validity labels. See the
[limitations statement](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md) before
making a scientific claim.
