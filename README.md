# PDECert

Symbolic PDE candidates should come with a certificate or a concrete counterexample.

PDECert is an early verifier for analytical PDE solution candidates. It checks
the PDE residual together with initial and boundary conditions, looks for domain
singularities, and returns one of three outcomes:

- `PROVED` when every current obligation is an exact symbolic identity;
- `REFUTED` when it finds a singularity or a numerical counterexample;
- `INCONCLUSIVE` when the available checks cannot decide.

Numerical sampling is only used to refute. Passing sampled points is never
treated as a proof.

> [!IMPORTANT]
> This is a research prototype, not a general theorem prover. The current
> `PROVED` result applies only to the obligations and domain checks represented
> by the input problem.

## Why this exists

PDE solvers and language models can produce expressions that look convincing
and have a small residual on a fixed grid. That does not guarantee that they
satisfy the full problem. A candidate can fail at a boundary, hide a pole
between collocation points, or work only at one parameter value.

The included experiment contains seven small heat-equation cases. Among the
five deliberately wrong candidates, the initial results are:

| Check | Wrong candidates accepted as valid |
| --- | ---: |
| PDE-only fixed collocation | 5 / 5 |
| Fixed collocation with initial/boundary conditions | 4 / 5 |
| SymPy PDE-only candidate check | 1 / 5 |
| PDECert | 0 / 5 |

PDECert refutes four and leaves the below-tolerance case `INCONCLUSIVE`. This is
a small adversarial experiment, not yet evidence of broad benchmark performance.

## Install and run

```bash
git clone https://github.com/oroikono/PDECert.git
cd PDECert
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m experiments.adversarial_heat
python -m experiments.sigs_poisson_gauss
```

The installed command accepts one versioned JSON case and prints a stable JSON
report:

```bash
pdecert verify examples/exact_heat.json
pdecert verify examples/exact_heat.json --output report.json
```

Exit code `0` means `PROVED`, `1` means `REFUTED`, `2` means `INCONCLUSIVE`,
and `64` reports an unreadable or invalid input file. A non-zero result is not
automatically a software failure; consumers should also read `report.status`.

## Small example

```python
import sympy as sp

from pdecert import Constraint, Problem, verify

x, t = sp.symbols("x t", real=True)
candidate = sp.exp(-sp.pi**2 * t) * sp.sin(sp.pi * x)

problem = Problem(
    name="heat equation",
    variables=(x, t),
    domains={x: (0.0, 1.0), t: (0.0, 1.0)},
    pde_residuals=(
        Constraint("PDE", sp.diff(candidate, t) - sp.diff(candidate, x, 2)),
    ),
    conditions=(
        Constraint("initial condition", candidate.subs(t, 0) - sp.sin(sp.pi * x)),
        Constraint("left boundary", candidate.subs(x, 0)),
        Constraint("right boundary", candidate.subs(x, 1)),
    ),
)

report = verify(problem, (candidate,))
print(report.status)  # Status.PROVED
```

## JSON cases

Version 1 of the case format stores a fully instantiated problem: declared real
variables, rectangular domains, residual expressions, conditions, and the
candidate expressions used for domain checks. The canonical shape is defined in
[`schema/problem-v1.schema.json`](schema/problem-v1.schema.json), with a complete
example in [`examples/exact_heat.json`](examples/exact_heat.json).

```python
from pdecert import load_case, verify

case = load_case("examples/exact_heat.json")
report = verify(case.problem, case.candidate_expressions)
print(report.to_dict())
```

Expression strings use a deliberately restricted arithmetic grammar. Declared
variables, numeric literals, `pi`, `E`, and a documented set of SymPy functions
are accepted. Attribute access, imports, indexing, unknown names, and keyword
arguments are rejected before parsing. This first schema represents residuals
after the candidate has been substituted; an operator-level PDE schema is a
later milestone.

## Current limits

The prototype does not yet define weak or viscosity solution semantics. It also
needs resource limits for symbolic simplification, stronger multivariate domain
analysis, interval arithmetic, and supported a posteriori error bounds. When a
check is incomplete, the intended behavior is `INCONCLUSIVE`.

The next milestones are tracked in [ROADMAP.md](ROADMAP.md). Contributions that
add one focused capability together with tests are welcome.

## License

MIT
