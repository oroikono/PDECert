import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.train_fisher_kpp_pinn import (
    _build_dense_tanh_model,
    _loss_components,
    build_manifest,
    main,
    train,
)


class FisherKppTrainingContractTests(unittest.TestCase):
    def test_artifact_id_uses_the_declared_seed(self):
        state = {
            "0.bias": [0.0, 0.0],
            "0.weight": [[1.0, 0.0], [0.0, 1.0]],
            "2.bias": [0.0],
            "2.weight": [[1.0, -1.0]],
        }
        manifest = build_manifest(
            torch=SimpleNamespace(__version__="test"),
            state_dict=state,
            final_losses={"pde_mse": 1.0},
            seed=17,
            steps=1,
            hidden_widths=[2],
            learning_rate=0.001,
        )
        self.assertEqual(manifest["artifact_id"], "fisher-kpp-pinn-seed-17")
        self.assertEqual(manifest["training"]["seed"], 17)

    def test_cli_rejects_architectures_outside_the_frozen_contract_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.json"
            for option, value in (("--width", "129"), ("--hidden-layers", "5")):
                with self.subTest(option=option):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(["--output", str(output), "--steps", "1", option, value])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertFalse(output.exists())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "training regression requires the optional PyTorch dependency")
class FisherKppTrainingLossTests(unittest.TestCase):
    def test_final_losses_are_evaluated_at_the_frozen_weights(self):
        torch_module, state, final_losses = train(
            seed=17,
            steps=1,
            hidden_widths=[2],
            learning_rate=0.001,
        )
        model = _build_dense_tanh_model(torch_module, [2]).to(dtype=torch.float64, device="cpu")
        model.load_state_dict(
            {
                name: torch.tensor(value, dtype=torch.float64, device="cpu")
                for name, value in state.items()
            }
        )
        x_axis = torch.linspace(-6.0, 6.0, 17, dtype=torch.float64)
        t_axis = torch.linspace(0.0, 2.0, 17, dtype=torch.float64)
        x_mesh, t_mesh = torch.meshgrid(x_axis, t_axis, indexing="ij")
        collocation = torch.stack((x_mesh.reshape(-1), t_mesh.reshape(-1)), dim=1)
        initial_x = torch.linspace(-6.0, 6.0, 33, dtype=torch.float64).reshape(-1, 1)
        boundary_t = torch.linspace(0.0, 2.0, 33, dtype=torch.float64).reshape(-1, 1)
        pde, initial, boundary, total = _loss_components(
            torch_module,
            model,
            collocation,
            initial_x,
            boundary_t,
        )
        observed = {
            "boundary_mse": float(boundary.detach()),
            "initial_mse": float(initial.detach()),
            "pde_mse": float(pde.detach()),
            "weighted_total": float(total.detach()),
        }
        self.assertEqual(final_losses, observed)


if __name__ == "__main__":
    unittest.main()
