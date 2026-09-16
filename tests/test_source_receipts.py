import builtins
import copy
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema

from pdecert import FrozenCallableError, validate_frozen_callable_integrity
from pdecert.source_receipts import (
    capture_source_receipt,
    require_active_source_root,
    validate_source_receipt,
)
from experiments import current_fisher_kpp_pair as current
from experiments import trained_fisher_kpp_pair as historical


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / current.DEFAULT_HISTORY


class HistoricalSourceTests(unittest.TestCase):
    def test_snapshot_matches_every_original_digest_and_has_only_declared_sources(self):
        integrity = validate_frozen_callable_integrity(
            historical.DEFAULT_FIXTURE,
            historical.DEFAULT_INTEGRITY,
            historical_source_root=HISTORY,
        )
        sources = integrity["source_files_sha256"]
        self.assertEqual(len(sources), 18)
        observed = {
            path.relative_to(HISTORY).as_posix()
            for path in HISTORY.rglob("*")
            if path.is_file() and path.name not in {"README.md", "snapshot.json"}
        }
        self.assertEqual(observed, set(sources))
        manifest = json.loads((HISTORY / "snapshot.json").read_text())
        self.assertEqual(manifest["source_files_sha256"], sources)
        self.assertEqual(
            manifest["snapshot_source_revision"], "c962d9c2df4a37bd682fd187e739c8db0edfb1d5"
        )

    def test_default_validation_still_rejects_changed_current_sources(self):
        with self.assertRaisesRegex(FrozenCallableError, "source_files_sha256.*digest mismatch"):
            validate_frozen_callable_integrity(
                historical.DEFAULT_FIXTURE, historical.DEFAULT_INTEGRITY
            )

    def test_each_tampered_or_missing_historical_source_is_rejected(self):
        sources = json.loads(historical.DEFAULT_INTEGRITY.read_text())["source_files_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "historical"
            shutil.copytree(HISTORY, copied)
            for relative in sources:
                with self.subTest(relative=relative):
                    path = copied / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# changed")
                    with self.assertRaisesRegex(FrozenCallableError, "digest mismatch"):
                        validate_frozen_callable_integrity(
                            historical.DEFAULT_FIXTURE,
                            historical.DEFAULT_INTEGRITY,
                            historical_source_root=copied,
                        )
                    path.unlink()
                    with self.assertRaisesRegex(FrozenCallableError, "file does not exist"):
                        validate_frozen_callable_integrity(
                            historical.DEFAULT_FIXTURE,
                            historical.DEFAULT_INTEGRITY,
                            historical_source_root=copied,
                        )
                    path.write_bytes(original)

    def test_history_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "historical"
            shutil.copytree(HISTORY, copied)
            source = copied / "src/pdecert/core.py"
            source.unlink()
            source.symlink_to(HISTORY / "src/pdecert/core.py")
            with self.assertRaisesRegex(FrozenCallableError, "escapes"):
                validate_frozen_callable_integrity(
                    historical.DEFAULT_FIXTURE,
                    historical.DEFAULT_INTEGRITY,
                    historical_source_root=copied,
                )


class CurrentSourceReceiptTests(unittest.TestCase):
    def test_complete_active_source_receipt_conforms_to_public_schema(self):
        receipt = capture_source_receipt(ROOT, runner_paths=current.RUNNER_PATHS)
        schema = json.loads((ROOT / "schema/source-receipt-v1.schema.json").read_text())
        jsonschema.validate(receipt, schema)
        validate_source_receipt(receipt, ROOT, runner_paths=current.RUNNER_PATHS)
        self.assertIn("src/pdecert/core.py", receipt["source_files_sha256"])
        self.assertIn("src/pdecert/source_receipts.py", receipt["source_files_sha256"])
        require_active_source_root(ROOT, (current.__file__, historical.__file__))

    def test_source_tamper_unbound_files_and_wrong_versions_are_rejected(self):
        receipt = capture_source_receipt(ROOT, runner_paths=current.RUNNER_PATHS)
        for field in ("schema_version", "integrity_scope", "runner_paths", "source_files_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(receipt)
                if field == "schema_version":
                    changed[field] = True
                elif field == "integrity_scope":
                    changed[field] = "proof"
                elif field == "runner_paths":
                    changed[field] = []
                else:
                    changed[field].pop("src/pdecert/core.py")
                with self.assertRaises(FrozenCallableError):
                    validate_source_receipt(changed, ROOT, runner_paths=current.RUNNER_PATHS)
        receipt["source_files_sha256"]["src/pdecert/core.py"] = "0" * 64
        with self.assertRaisesRegex(FrozenCallableError, "digest mismatch"):
            validate_source_receipt(receipt, ROOT, runner_paths=current.RUNNER_PATHS)

    def test_addition_change_and_missing_runner_in_a_copied_source_tree_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/pdecert"
            package.mkdir(parents=True)
            (package / "source_receipts.py").write_text("# synthetic source\n")
            (root / "runner.py").write_text("# synthetic runner\n")
            (root / "pyproject.toml").write_text("# synthetic metadata\n")
            receipt = capture_source_receipt(root, runner_paths=["runner.py"])
            (package / "new.py").write_text("# newly introduced source\n")
            with self.assertRaisesRegex(FrozenCallableError, "inventory"):
                validate_source_receipt(receipt, root, runner_paths=["runner.py"])
            (package / "new.py").unlink()
            (root / "runner.py").write_text("# changed\n")
            with self.assertRaisesRegex(FrozenCallableError, "digest mismatch"):
                validate_source_receipt(receipt, root, runner_paths=["runner.py"])
            (root / "runner.py").unlink()
            with self.assertRaisesRegex(FrozenCallableError, "does not exist"):
                validate_source_receipt(receipt, root, runner_paths=["runner.py"])

    def test_runner_paths_reject_traversal_and_claiming_another_active_checkout(self):
        for path in ("../runner.py", "/tmp/runner.py", "experiments/../runner.py"):
            with self.subTest(path=path), self.assertRaises(FrozenCallableError):
                capture_source_receipt(ROOT, runner_paths=[path])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FrozenCallableError):
                require_active_source_root(directory, (current.__file__,))


class CurrentFisherKppReceiptTests(unittest.TestCase):
    def test_reduced_historical_inventory_is_rejected_before_validation_or_materialization(self):
        integrity = json.loads(historical.DEFAULT_INTEGRITY.read_text())
        integrity["source_files_sha256"].pop("src/pdecert/autodiff.py")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            copied = Path(directory) / "integrity.json"
            copied.write_text(json.dumps(integrity))
            with (
                patch.object(
                    current,
                    "validate_frozen_callable_integrity",
                    wraps=validate_frozen_callable_integrity,
                ) as validate,
                patch.object(
                    historical,
                    "build_case",
                    side_effect=AssertionError("invalid history must not materialize the model"),
                ) as materialize,
            ):
                for evaluate in (False, True):
                    with (
                        self.subTest(evaluate=evaluate),
                        self.assertRaisesRegex(FrozenCallableError, "historical integrity digest"),
                    ):
                        current.run(
                            evaluate=evaluate, historical_source_root=HISTORY, integrity=copied
                        )
                validate.assert_not_called()
                materialize.assert_not_called()

    def test_byte_identical_integrity_copy_preserves_all_source_bindings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            copied = Path(directory) / "integrity.json"
            copied.write_bytes(historical.DEFAULT_INTEGRITY.read_bytes())
            receipt = current.run(historical_source_root=HISTORY, integrity=copied)
            self.assertEqual(
                receipt["historical_integrity"]["source_files_sha256"],
                json.loads(historical.DEFAULT_INTEGRITY.read_text())["source_files_sha256"],
            )
            self.assertEqual(
                receipt["historical_integrity"]["file"]["path"],
                copied.relative_to(ROOT).as_posix(),
            )
            current.validate_receipt(receipt, historical_source_root=HISTORY, integrity=copied)

    def test_inspection_checks_identity_without_torch_or_evaluation(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("identity inspection must not import torch")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded),
            patch.object(historical, "build_case", side_effect=AssertionError("must not evaluate")),
        ):
            receipt = current.run(historical_source_root=HISTORY)
        self.assertEqual(receipt["operation"], "content_identity_check")
        self.assertIsNone(receipt["matched_report"])
        self.assertFalse(receipt["historical_replay"])
        self.assertNotEqual(
            receipt["historical_integrity"]["source_files_sha256"][
                "src/pdecert/frozen_callable.py"
            ],
            receipt["current_evaluator"]["source_files_sha256"]["src/pdecert/frozen_callable.py"],
        )
        current.validate_receipt(receipt, historical_source_root=HISTORY)

    def test_every_unbound_active_problem_input_is_rejected_even_with_identical_bytes(self):
        for label, source in (
            ("template", historical.DEFAULT_TEMPLATE),
            ("raw", historical.DEFAULT_RAW),
            ("case", historical.DEFAULT_CASE),
            ("record", historical.DEFAULT_RECORD),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                copied = Path(directory) / source.name
                copied.write_bytes(source.read_bytes())
                with self.assertRaisesRegex(FrozenCallableError, "not bound"):
                    current.run(historical_source_root=HISTORY, **{label: copied})

    def test_receipt_changes_to_history_inputs_report_and_mode_are_rejected(self):
        receipt = current.run(historical_source_root=HISTORY)
        for field in (
            "active_inputs",
            "historical_integrity",
            "matched_report",
            "historical_replay",
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(receipt)
                changed[field] = True if field == "historical_replay" else {}
                with self.assertRaises(FrozenCallableError):
                    current.validate_receipt(changed, historical_source_root=HISTORY)
        evaluated = copy.deepcopy(receipt)
        # A synthetic report only exercises receipt hashing; it is not a numerical run.
        evaluated["operation"] = "current_evaluation"
        evaluated["invocation"]["evaluate"] = True
        evaluated["matched_report"] = {"synthetic_test": True}
        evaluated["matched_report_sha256"] = current._canonical_sha256(evaluated["matched_report"])
        current.validate_receipt(evaluated, historical_source_root=HISTORY)
        evaluated["matched_report"]["synthetic_test"] = False
        with self.assertRaisesRegex(FrozenCallableError, "matched_report_sha256"):
            current.validate_receipt(evaluated, historical_source_root=HISTORY)

    def test_sources_changing_during_an_operation_are_rejected(self):
        original = current.inspect_inputs(historical_source_root=HISTORY)
        changed = copy.deepcopy(original)
        changed["current_evaluator"]["source_files_sha256"]["src/pdecert/core.py"] = "0" * 64
        with patch.object(current, "inspect_inputs", side_effect=[original, changed]):
            with self.assertRaisesRegex(FrozenCallableError, "changed during"):
                current.run(historical_source_root=HISTORY)

    def test_runtime_invocation_and_date_metadata_reject_inconsistent_shapes(self):
        receipt = current.run(historical_source_root=HISTORY)
        mutations = (
            ("runtime", {"python": True}),
            ("git", {"head_revision": "unknown"}),
            ("created_at", "2026-09-15"),
            ("invocation", {"evaluate": False}),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(receipt)
                changed[field] = value
                with self.assertRaises(FrozenCallableError):
                    current.validate_receipt(changed, historical_source_root=HISTORY)

    def test_explicit_evaluation_dispatch_and_configuration_with_a_synthetic_report(self):
        # Mock dispatch verifies wiring only; it is not a real PyTorch reproduction.
        synthetic = {"synthetic_test": True}
        with (
            patch.object(historical, "build_case", return_value=("case", None, None)),
            patch.object(
                current,
                "verify_matched_case",
                return_value=SimpleNamespace(to_dict=lambda: synthetic),
            ) as evaluate,
        ):
            receipt = current.run(evaluate=True, historical_source_root=HISTORY)
        self.assertEqual(evaluate.call_args.args, ("case",))
        options = evaluate.call_args.kwargs["options"]
        self.assertEqual(options["trained-pinn"].tolerance, 1e-3)
        self.assertEqual(options["trained-pinn"].samples_per_axis, 6)
        self.assertEqual(options["symbolic-qwen3"].symbolic_timeout, 2.0)
        self.assertEqual(receipt["operation"], "current_evaluation")
        self.assertEqual(receipt["matched_report"], synthetic)
        current.validate_receipt(receipt, historical_source_root=HISTORY)
        with self.assertRaisesRegex(FrozenCallableError, "boolean"):
            current.run(evaluate="yes", historical_source_root=HISTORY)

    def test_cli_validates_receipts_and_refuses_to_replace_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            args = ["--historical-source-root", str(HISTORY), "--output", str(output)]
            self.assertEqual(current.main(args), 0)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                current.main(args)
            self.assertEqual(output.read_bytes(), original)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    current.main(
                        [
                            "--historical-source-root",
                            str(HISTORY),
                            "--validate",
                            str(output),
                        ]
                    ),
                    0,
                )
