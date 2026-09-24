"""Exercise the saved-sample guide with source and core-only installed packages."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "reference-files.md"


class ReferenceWalkthroughTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guide = GUIDE.read_text(encoding="utf-8")
        blocks = re.findall(
            r"<!-- tested-reference-walkthrough:start -->\n(.*?)"
            r"<!-- tested-reference-walkthrough:end -->",
            guide,
            re.DOTALL,
        )
        if len(blocks) != 1:
            raise AssertionError("expected exactly one tested reference walkthrough")
        recipe = re.fullmatch(
            r"```bash\npython - <<'PY' &&\n(.*?)\nPY\n"
            r"(python -m pdecert\.cli [^\n]+) &&\n"
            r"python - <<'PY'\n(.*?)\nPY\n```\n",
            blocks[0],
            re.DOTALL,
        )
        if recipe is None:
            raise AssertionError("expected input creation, CLI comparison, and report reading")
        cls.create_source, command, cls.read_source = recipe.groups()
        cls.compare_arguments = shlex.split(command)[1:]

    def run_python(self, directory, arguments):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-B"]
        if sys.flags.isolated:
            command.append("-I")
            env.pop("PYTHONPATH", None)
        elif "PYTHONPATH" in env:
            env["PYTHONPATH"] = os.pathsep.join(
                str(Path(entry).resolve()) for entry in env["PYTHONPATH"].split(os.pathsep)
            )
        return subprocess.run(
            command + arguments,
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def run_recipe(self, updates=None):
        with tempfile.TemporaryDirectory() as directory:
            created = self.run_python(directory, ["-c", self.create_source])
            self.assertEqual(created.returncode, 0, created.stderr)
            input_path = Path(directory) / "samples.json"
            if updates:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
                payload.update(updates)
                input_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            original = input_path.read_bytes()
            compared = self.run_python(directory, self.compare_arguments)
            self.assertEqual(compared.stdout, "")
            self.assertEqual(input_path.read_bytes(), original)
            output_path = Path(directory) / "comparison.json"
            if compared.returncode:
                self.assertFalse(output_path.exists())
                return compared, None
            self.assertEqual(compared.stderr, "")
            report = json.loads(output_path.read_text(encoding="utf-8"))
            displayed = self.run_python(directory, ["-c", self.read_source])
            self.assertEqual(displayed.returncode, 0, displayed.stderr)
            self.assertEqual(displayed.stderr, "")
            self.assertEqual(json.loads(displayed.stdout), report["fields"][0])
            self.assertEqual(report["evidence_level"], "EMPIRICAL")
            self.assertNotIn("status", report)
            return compared, report

    def successful_report(self, updates=None):
        result, report = self.run_recipe(updates)
        self.assertEqual(result.returncode, 0, result.stderr)
        return report

    def test_verbatim_recipe_compares_its_own_saved_values(self):
        report = self.successful_report()
        self.assertEqual(report["sample_count"], 3)
        field = report["fields"][0]
        self.assertAlmostEqual(field["rmse"], 0.5773502691896257)
        self.assertAlmostEqual(field["relative_l2"], 0.26726124191242434)
        self.assertEqual(field["max_absolute_error"], 1.0)
        self.assertEqual(field["max_error_sample"]["coordinates"], {"x": 1.0})

    def test_matching_values_remain_empirical(self):
        report = self.successful_report({"candidate": {"u": [1.0, 2.0, 3.0]}})
        field = report["fields"][0]
        for metric in ("rmse", "relative_l2", "max_absolute_error"):
            self.assertEqual(field[metric], 0.0)

    def test_zero_reference_norm_is_unavailable_not_a_failed_command(self):
        report = self.successful_report(
            {"candidate": {"u": [0.0, 0.0, 0.0]}, "reference": {"u": [0.0, 0.0, 0.0]}}
        )
        field = report["fields"][0]
        self.assertEqual(field["rmse"], 0.0)
        self.assertIsNone(field["relative_l2"])
        self.assertEqual(field["relative_l2_unavailable_reason"], "zero_reference_norm")

    def test_misaligned_columns_are_rejected_without_a_report(self):
        result, _ = self.run_recipe({"candidate": {"u": [1.0, 2.0]}})
        self.assertEqual(result.returncode, 64)
        self.assertIn("expected 3 aligned samples", result.stderr)

    def test_unsupported_version_is_rejected_without_a_report(self):
        result, _ = self.run_recipe({"reference_input_version": 2})
        self.assertEqual(result.returncode, 64)
        self.assertIn("reference_input_version", result.stderr)

    def test_creation_does_not_replace_an_existing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "samples.json"
            input_path.write_text("keep these original samples", encoding="utf-8")
            result = self.run_python(directory, ["-c", self.create_source])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(input_path.read_text(encoding="utf-8"), "keep these original samples")


if __name__ == "__main__":
    unittest.main()
