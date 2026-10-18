"""Exercise the saved trained-model example from outside the checkout."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pdecert import report_from_dict


def test_saved_trained_model_audit_runs_without_retraining(tmp_path):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    example = root / "examples/audit_trained_model.py"
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    audit = json.loads(result.stdout)
    assert "status" not in audit
    assert audit["evaluation"]["training_performed"] is False
    assert audit["evaluation"]["total_sample_points"] == 54
    assert len(audit["obligations"]) == 4
    reports = [report_from_dict(item["diagnostic_report"]) for item in audit["obligations"]]
    assert reports[0].status.value == "REFUTED"
    assert all(report.status.value != "PROVED" for report in reports)
    fixture = root / "benchmarks/matched/fisher-kpp-classical-01/pinn.json"
    provenance = audit["candidate"]["provenance"]
    assert provenance["artifact_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert provenance["wrapper_sha256"] == hashlib.sha256(example.read_bytes()).hexdigest()
