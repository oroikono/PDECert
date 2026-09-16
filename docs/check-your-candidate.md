# Check your own symbolic candidate

This is the next step after the [offline quickstart](quickstart.md): supply an
expression yourself and inspect its evidence. You only need the core package,
not a repository checkout, PyTorch, an agent framework, or model credentials.

## Keep the problem separate from the answer

The example checks the classical heat equation `u_t - u_xx = 0` on
`x in [0, 1]`, `t in [0, 1]`, with initial data `u(x, 0) = sin(pi*x)` and
boundary data `u(0, t) = u(1, t) = 0`.

The template below is the trusted problem. To try a different candidate for
**this same problem**, replace only the quoted expression after `python -`.
Do not let a candidate generator change the equation or conditions it is being
checked against.

Run this block in a terminal with Python 3.10–3.14 on macOS or Linux, preferably
in a virtual environment. Installation needs network access; the check itself
runs locally without network or model calls.

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

The first result has `report.status: "PROVED"` and
`report.decision_evidence: "EXACT"`. That establishes the represented symbolic
obligations for this expression, not existence, uniqueness, or a proof checked
by a formal proof-assistant kernel.

## Change just the candidate

Rerun the block with each of these expressions in the quoted argument:

| Expression | Expected result | What it demonstrates |
| --- | --- | --- |
| `exp(-pi**2*t)*sin(pi*x)` | `PROVED`, `EXACT` | The equation and represented conditions hold exactly. |
| `exp(-pi**2*t)*sin(pi*x) + x/10` | `REFUTED`, `EMPIRICAL` | The PDE still holds, but the initial and right boundary data are wrong. |
| `exp(-pi**2*t)*sin(pi*x) + t*x*(1-x)/10**14` | `INCONCLUSIVE` | A nonzero error passes the numerical tolerance; that does not prove correctness. |
| `sin(y)` | `TemplateError` | An undeclared variable is rejected before verification. |

For the `+ x/10` case, the first witness is the **initial condition**, at
`x = 0.113`, with residual approximately `0.0113`. Here `t = 0` has already
been substituted by `At(u, t, 0)`, so it is not repeated in the witness's `point`.
The report need not list every violation after it has found one.

Read `report.witness` for the failing obligation and point,
`report.evidence_events` for the supporting evidence, and
`report.incomplete_reasons` when the answer is inconclusive. A floating-point
counterexample is empirical; replay numerically sensitive findings at higher
precision or independently before making a mathematical claim.

This Python snippet prints a report and normally exits successfully for all
three statuses. Inspect `report.status`; do not interpret shell success as a
proof. The separate [`pdecert verify` CLI](../README.md#install-and-run) has
status-specific exit codes.

## Use a different PDE

As the problem author, replace the template's variables, domains, operators,
and conditions together, then supply a separate candidate. `D(u, x, 2)` means
the second derivative of `u` with respect to `x`; `At(u, x, 0)` substitutes a
boundary coordinate. See the [template contract](problem-templates.md) for the
accepted grammar and field-binding rules.

This path accepts symbolic expressions for classical strong-form problems on
rectangular domains. It does not accept arbitrary Python programs, neural
checkpoints, weak solutions, or general meshes. Unsupported syntax is an input
error, not a mathematical refutation. Do not use `eval` to turn model text into
a candidate.

The two-second deadline applies to individual symbolic checks on supported
main-thread interval-timer environments. It does not bound parsing, all work,
or process memory. If deadlines are unavailable or a check times out, the
verifier abstains unless another check can refute. Do not expose this local
recipe as a public service for untrusted workloads without outer process and
resource isolation. See [limitations](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md).

## Keep and contribute useful evidence

The JSON includes the instantiated case, evaluator settings, and complete
report. Retain the original candidate text and your package/runtime versions
alongside it; this wrapper is not a digest-bound run manifest or a human label.
For publication, follow the [run-manifest contract](run-manifests.md).

Have a real candidate that exposes an interesting failure? Use the
[failure-case form](https://github.com/oroikono/PDECert/issues/new?template=failure-case.yml)
with the unchanged output, problem, provenance, and report. The three examples
above are deliberately constructed teaching cases, not new natural benchmark
records.
