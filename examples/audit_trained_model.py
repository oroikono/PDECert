"""Audit a bundled trained model without training or changing its weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pdecert import (
    audit_callable,
    load_frozen_callable,
    load_template,
    materialize_frozen_callable,
)


def run() -> dict[str, object]:
    """Use the saved Fisher--KPP PINN as a small, offline own-model example."""
    root = Path(__file__).resolve().parents[1]
    fixture = root / "benchmarks/matched/fisher-kpp-classical-01/pinn.json"
    template = load_template(fixture.with_name("template.json"))
    frozen = load_frozen_callable(fixture)
    if frozen.input_names != template.variables or frozen.output_names != template.field_names:
        raise ValueError("model coordinate and field order must match the template")
    candidate = materialize_frozen_callable(frozen)
    return audit_callable(
        template,
        candidate,
        candidate_id=frozen.artifact_id,
        provenance={
            "artifact_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "description": "Bundled Fisher--KPP PINN; existing weights, no retraining",
        },
        tolerance=1e-3,
        samples_per_axis=6,
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
