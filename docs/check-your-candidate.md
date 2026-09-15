# Check your own symbolic candidate

Check an expression against a heat-equation problem, then change it to see how
the result changes. This example uses only the core package. It works without
a repository checkout or optional dependencies.

## Run the example

The problem is `u_t - u_xx = 0` on `x in [0, 1]`, `t in [0, 1]`, with
`u(x, 0) = sin(pi*x)` and `u(0, t) = u(1, t) = 0`. The template below defines
that problem; the quoted expression after `python -` is the proposed solution.

Paste the whole block into a terminal on macOS or Linux with Python 3.10–3.14,
preferably in the virtual environment from the [quickstart](quickstart.md).
Installation needs network access. The check runs locally without network or
model calls.

<!-- tested-candidate-walkthrough:start -->
```bash
python -m pip install pdecert==0.1.1rc2
python - 'exp(-pi**2*t)*sin(pi*x)' <<'PY'
import json
import sys

from pdecert import bind_symbolic_candidate, case_to_dict, template_from_dict, verify

template = template_from_dict({
    "template_version": 1,
    "name": "heat equation",
    "solution_semantics": "classical_strong",
    "variables": ["x", "t"],
    "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
    "parameters": {},
    "field_names": ["u"],
    "pde_residuals": [
        {"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"},
    ],
    "conditions": [
        {"name": "initial condition", "expression": "At(u, t, 0) - sin(pi*x)"},
        {"name": "left boundary", "expression": "At(u, x, 0)"},
        {"name": "right boundary", "expression": "At(u, x, 1)"},
    ],
})
settings = {
    "tolerance": 1e-9,
    "samples_per_axis": 5,
    "symbolic_timeout": 2.0,
    "max_expression_ops": 10000,
}
case = bind_symbolic_candidate(template, {"u": sys.argv[1]})
report = verify(case.problem, case.candidate_fields, **settings)
print(json.dumps({
    "case": case_to_dict(case),
    "settings": settings,
    "report": report.to_dict(),
}, indent=2, allow_nan=False))
PY
```
<!-- tested-candidate-walkthrough:end -->

In the printed JSON, look for `report.status: "PROVED"` and
`report.decision_evidence: "EXACT"`. The expression passes the encoded equation,
conditions, and domain checks through exact symbolic evidence. This result does
not establish existence or uniqueness, or supply a proof checked by a formal
proof-assistant kernel.

## Try another expression

Replace the quoted expression on the `python -` line with one from the table
and rerun the Python block. You only need to install the package once.
Keep the template fixed when comparing candidates: a generator must not change
the equation or conditions its answer is checked against.

| Expression | Expected result | Why |
| --- | --- | --- |
| `exp(-pi**2*t)*sin(pi*x)` | `PROVED`, `EXACT` | The encoded equation and conditions hold exactly. |
| `exp(-pi**2*t)*sin(pi*x) + x/10` | `REFUTED`, `EMPIRICAL` | The PDE still holds, but the initial and right boundary data are wrong. |
| `exp(-pi**2*t)*sin(pi*x) + t*x*(1-x)/10**14` | `INCONCLUSIVE` | A nonzero residual falls below the numerical tolerance. Passing samples cannot prove correctness. |
| `sin(y)` | `TemplateError` | An undeclared variable is rejected before verification. |

## Read the report

For a `REFUTED` result, start with `report.witness`. In the `+ x/10` example,
it identifies the initial condition at `x = 0.113`, with residual approximately
`0.0113`. The template already substituted `t = 0`, so the witness's `point`
only contains `x`. The verifier can stop after finding one violation.

For `INCONCLUSIVE`, read `report.incomplete_reasons` to see what the verifier
could not establish. `report.evidence_events` records the evidence for each
check. Floating-point counterexamples are empirical; replay numerically
sensitive findings at higher precision or independently before making a
mathematical claim.

The Python block exits successfully for all three statuses, so read
`report.status` to learn the result. The separate
[`pdecert verify` CLI](../README.md#install-and-run) uses status-specific exit
codes.

## Use a different PDE

To check your own PDE, define its variables, domains, operators, and conditions
in the template, then supply a candidate separately. `D(u, x, 2)` is the second
derivative of `u` with respect to `x`; `At(u, x, 0)` substitutes `x = 0`.
The [template guide](problem-templates.md) explains the accepted grammar and
how expressions bind to fields.

This path uses pointwise derivatives (`classical_strong` semantics) on
rectangular domains. It accepts symbolic expressions, but not arbitrary Python
programs, neural checkpoints, weak solutions, or general meshes. Unsupported
syntax is an input error, not a mathematical refutation. Do not use `eval` to
turn model text into a candidate.

The two-second timeout limits individual symbolic checks on supported
main-thread interval-timer environments. It does not limit parsing, total
runtime, or process memory. If the timeout is unavailable or reached, the
verifier remains inconclusive unless another check can refute. A public service
for untrusted workloads needs outer process and resource isolation. See the
[limitations](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md).

## Save or share a result

Save the JSON together with the original candidate text and your package and
Python versions. The JSON contains the case, settings, and complete report; it
is not a digest-bound run manifest or a human label. For publication, follow
the [run-manifest guide](run-manifests.md).

To contribute a real candidate that fails its stated problem, use the
[failure-case form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml)
with the unchanged output, problem, provenance, and report. The expressions
above are constructed teaching cases, not natural benchmark records.
