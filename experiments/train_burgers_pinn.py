"""Train the frozen nonlinear Burgers PINN fixture on deterministic CPU points."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from experiments.burgers_pinn_fixture import (
    build_dense_tanh_model,
    canonical_weights_sha256,
    write_new_manifest,
)


VISCOSITY = 0.1
WAVE_SPEED = 0.5


def _derivative(torch, value, points, column):
    gradient = torch.autograd.grad(
        value,
        points,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient[:, column : column + 1]


def _exact_trace(torch, x, t):
    return WAVE_SPEED - torch.tanh((x - WAVE_SPEED * t) / (2.0 * VISCOSITY))


def train(*, seed: int, steps: int, hidden_widths: list[int], learning_rate: float):
    """Train one PINN only from the PDE and declared initial/boundary traces."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to reproduce the trained fixture") from error

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = build_dense_tanh_model(torch, hidden_widths).to(dtype=torch.float64, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_axis = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    t_axis = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    x_mesh, t_mesh = torch.meshgrid(x_axis, t_axis, indexing="ij")
    collocation = torch.stack((x_mesh.reshape(-1), t_mesh.reshape(-1)), dim=1)
    initial_x = torch.linspace(-1.0, 1.0, 33, dtype=torch.float64).reshape(-1, 1)
    boundary_t = torch.linspace(0.0, 1.0, 33, dtype=torch.float64).reshape(-1, 1)

    final_losses: dict[str, float] = {}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        points = collocation.detach().clone().requires_grad_(True)
        field = model(points)
        field_x = _derivative(torch, field, points, 0)
        field_t = _derivative(torch, field, points, 1)
        field_xx = _derivative(torch, field_x, points, 0)
        residual = field_t + field * field_x - VISCOSITY * field_xx
        pde_loss = torch.mean(residual**2)

        initial_points = torch.cat((initial_x, torch.zeros_like(initial_x)), dim=1)
        initial_target = _exact_trace(torch, initial_x, torch.zeros_like(initial_x))
        initial_loss = torch.mean((model(initial_points) - initial_target) ** 2)

        left_points = torch.cat((-torch.ones_like(boundary_t), boundary_t), dim=1)
        right_points = torch.cat((torch.ones_like(boundary_t), boundary_t), dim=1)
        left_target = _exact_trace(torch, -torch.ones_like(boundary_t), boundary_t)
        right_target = _exact_trace(torch, torch.ones_like(boundary_t), boundary_t)
        boundary_loss = torch.mean((model(left_points) - left_target) ** 2) + torch.mean(
            (model(right_points) - right_target) ** 2
        )

        loss = pde_loss + 10.0 * initial_loss + 10.0 * boundary_loss
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == steps:
            print(f"step={step} loss={loss.item():.6e} pde={pde_loss.item():.6e}")
        final_losses = {
            "boundary_mse": float(boundary_loss.detach()),
            "initial_mse": float(initial_loss.detach()),
            "pde_mse": float(pde_loss.detach()),
            "weighted_total": float(loss.detach()),
        }

    state_dict = {
        name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()
    }
    return torch, state_dict, final_losses


def build_manifest(
    *,
    torch,
    state_dict: dict[str, object],
    final_losses: dict[str, float],
    seed: int,
    steps: int,
    hidden_widths: list[int],
    learning_rate: float,
) -> dict[str, object]:
    """Bind weights to their architecture and complete training provenance."""

    script = Path(__file__)
    return {
        "schema_version": 1,
        "artifact_id": "burgers-pinn-seed-20260827",
        "artifact_kind": "trained_callable",
        "problem_id": "burgers-traveling-wave-classical-01",
        "architecture": {
            "type": "dense_mlp",
            "activation": "tanh",
            "dtype": "float64",
            "input_names": ["x", "t"],
            "hidden_widths": hidden_widths,
            "output_names": ["u"],
        },
        "training": {
            "method": "physics_informed_collocation",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "seed": seed,
            "steps": steps,
            "device": "cpu",
            "torch_version": torch.__version__,
            "collocation": {
                "pde_grid": {"x_points": 17, "t_points": 17},
                "initial_points": 33,
                "points_per_boundary": 33,
            },
            "loss_weights": {"pde": 1.0, "initial": 10.0, "boundary": 10.0},
            "final_losses": final_losses,
            "script": "experiments/train_burgers_pinn.py",
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "state_dict": state_dict,
        "weights_sha256": canonical_weights_sha256(state_dict),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=_positive_int, default=20260827)
    parser.add_argument("--steps", type=_positive_int, default=3000)
    parser.add_argument("--width", type=_positive_int, default=16)
    parser.add_argument("--hidden-layers", type=_positive_int, default=2)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.003)
    arguments = parser.parse_args(argv)
    hidden_widths = [arguments.width] * arguments.hidden_layers
    torch, state_dict, final_losses = train(
        seed=arguments.seed,
        steps=arguments.steps,
        hidden_widths=hidden_widths,
        learning_rate=arguments.learning_rate,
    )
    manifest = build_manifest(
        torch=torch,
        state_dict=state_dict,
        final_losses=final_losses,
        seed=arguments.seed,
        steps=arguments.steps,
        hidden_widths=hidden_widths,
        learning_rate=arguments.learning_rate,
    )
    write_new_manifest(arguments.output, manifest)
    print(f"Wrote trained callable fixture to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
