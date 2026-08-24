"""Verify exact and perturbed PyTorch heat-equation fields."""

from __future__ import annotations

import json

import torch

from pdecert import (
    AutodiffConstraint,
    AutodiffProblem,
    CallableCandidate,
    verify_artifact,
)


def build_problem() -> AutodiffProblem:
    """Return a heat problem with PDE, initial, and boundary obligations."""

    return AutodiffProblem(
        name="PyTorch heat equation",
        variables=("x", "t"),
        domains={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        pde_residuals=(
            AutodiffConstraint(
                "heat PDE",
                lambda value: value.derivative("u", "t") - value.derivative("u", "x", order=2),
            ),
        ),
        conditions=(
            AutodiffConstraint(
                "initial condition",
                lambda value: value.field("u") - torch.sin(torch.pi * value.coordinate("x")),
                fixed_coordinates={"t": 0.0},
            ),
            AutodiffConstraint(
                "left boundary",
                lambda value: value.field("u"),
                fixed_coordinates={"x": 0.0},
            ),
            AutodiffConstraint(
                "right boundary",
                lambda value: value.field("u"),
                fixed_coordinates={"x": 1.0},
            ),
        ),
    )


def exact_heat(points):
    """A differentiable exact heat-equation field."""

    x = points[:, 0:1]
    t = points[:, 1:2]
    return torch.exp(-(torch.pi**2) * t) * torch.sin(torch.pi * x)


def perturbed_heat(points):
    """A field that keeps the conditions but violates the PDE."""

    x = points[:, 0:1]
    t = points[:, 1:2]
    return exact_heat(points) + 0.1 * t * x * (1 - x)


def main() -> None:
    problem = build_problem()
    for name, field in (("exact", exact_heat), ("perturbed", perturbed_heat)):
        artifact = CallableCandidate.from_mapping({"u": field}, dtype="float64")
        report = verify_artifact(problem, artifact, tolerance=1e-9)
        print(json.dumps({"candidate": name, "report": report.to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
