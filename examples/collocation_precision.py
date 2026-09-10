"""Reproduce a cancellation-sensitive, empirical collocation failure.

This manufactured transport residual isolates numerical precision. It is not
a natural model output or a labeled Atlas record, and has no initial or boundary
conditions. Grid sampling does not certify the continuous domain.
"""

import json

from pdecert import FixedCollocationBaseline


def build_record() -> dict[str, object]:
    """Return the minimal symbolic input accepted by the baseline adapter."""

    return {
        "id": "transport-cancellation-example",
        "artifact_type": "symbolic_expression",
        "artifact": {"fields": {"u": "x**3/3 - 100000000*x**2 + 10000000000000001*x"}},
        "template": {
            "template_version": 1,
            "name": "cancellation-sensitive transport residual",
            "solution_semantics": "classical_strong",
            "variables": ["x", "t"],
            "domains": {"x": [100000000.0, 100000001.0], "t": [0.0, 1.0]},
            "parameters": {},
            "field_names": ["u"],
            "pde_residuals": [{"name": "transport PDE", "expression": "D(u, t) + D(u, x)"}],
            "conditions": [],
        },
    }


def run() -> dict[str, object]:
    """Evaluate at 30 digits without simplifying the residual to its stable form."""

    adapter = FixedCollocationBaseline(decimal_precision=30, points_per_axis=5, tolerance=1e-9)
    result = adapter.evaluate_record(build_record())
    assert result.outcome.value == "fail"
    assert result.max_absolute_residual == 2.0
    return {
        "configuration": dict(adapter.configuration()),
        "result": result.to_dict(),
        "scope": "Manufactured precision regression; empirical diagnostics, not a certificate.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
