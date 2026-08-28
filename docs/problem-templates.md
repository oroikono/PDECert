# Candidate-free problem templates

A `ProblemTemplate` stores the trusted mathematical task without storing any
candidate solution. It lets a symbolic expression, a trained PyTorch field, or
a future generator run target the same named variables, domains, fields, PDE
residuals, and conditions.

This separation matters for evaluation: the problem author controls the
obligations, while the candidate producer controls only the artifact being
checked. A model output cannot redefine its own boundary conditions.

## Version 1 contract

Version 1 accepts:

- scalar named fields over finite rectangular domains;
- the restricted expression grammar used by schema-v3 cases;
- classical strong, pointwise solution semantics;
- PDE residual, initial-condition, and boundary-condition expressions using
  arithmetic, supported scalar functions, `D`, and `At`.

The canonical JSON schema is
[`schema/problem-template-v1.schema.json`](../schema/problem-template-v1.schema.json).
Every declared field must occur in at least one trusted operator expression.
The format intentionally has no `fields` or `candidate_expressions` member.

Validate a saved template before publishing or using it:

```bash
pdecert template validate examples/heat-template.json
```

## Explicit symbolic binding

Binding requires exactly one expression for each declared field. Missing and
extra fields are errors, and the candidate expression is parsed through the
same non-executing restricted grammar as a normal case.

```python
from pdecert import bind_symbolic_candidate, load_template, verify

template = load_template("examples/heat-template.json")
case = bind_symbolic_candidate(
    template,
    {"u": "exp(-pi**2*t)*sin(pi*x)"},
)
report = verify(case.problem, case.candidate_fields)
assert report.status.value == "PROVED"
assert report.decision_evidence.value == "EXACT"
```

`template_from_case(case)` removes the candidate binding from an existing
schema-v3 case. Existing version 1, 2, and 3 case files remain readable; the
template format does not change their meaning.

## Callable evaluation from the same template

The operator compiler accepts a template directly. It compiles only the
trusted operator expressions. It does not translate, execute, or inspect a
candidate implementation.

```python
callable_problem = compile_autodiff_problem(template)
report = verify_artifact(callable_problem, callable_candidate)
```

Passing autodiff samples remains `INCONCLUSIVE`. A sampled violation may
produce `REFUTED` with a point witness, but empirical evaluation never inherits
exact evidence from a symbolic candidate bound to the same template.

## Unsupported inputs

Version 1 does not represent weak, entropy, or distributional solutions,
irregular geometry, interfaces, tensor-valued fields, or variational forms.
The current callable compiler also rejects parameter variables and expressions
outside its documented lowering subset. These boundaries are explicit errors,
not approximations.

## Reproduction

The symbolic path requires the core package. The callable example additionally
requires the `autodiff` extra:

```bash
pip install -e ".[dev,autodiff]"
pdecert template validate examples/heat-template.json
python -m examples.problem_template
```

The expected symbolic result is `PROVED (EXACT)`. The expected callable result
is `INCONCLUSIVE (sampled pass is not proof)`.
