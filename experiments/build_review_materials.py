"""Build a blank review form and a separate machine-assisted comparison file."""

from __future__ import annotations

import json
from pathlib import Path

from pdecert import load_corpus


INVALID_PROPOSALS = {
    "qwen-local-heat-mode-01": (
        ["pde_residual", "initial_condition", "boundary_condition"],
        "The candidate has no heat-equation time decay, adds a nonzero exponential trace, and fails the stated initial and boundary data.",
    ),
    "qwen-local-heat-mode-02": (
        ["pde_residual", "initial_condition"],
        "The spatial mode and decay rate do not match the second-mode heat problem, so both the PDE and initial trace fail.",
    ),
    "qwen-local-transport-01": (
        ["pde_residual"],
        "The static sine profile satisfies the initial trace but has nonzero spatial derivative in the transport residual.",
    ),
    "qwen-local-transport-02": (
        ["pde_residual"],
        "The static cosine profile satisfies the initial trace but omits translation at speed two.",
    ),
    "qwen-local-reaction-01": (
        ["pde_residual", "initial_condition"],
        "The added exponential term and stationary sine term do not satisfy u_t + u = 0 or the stated initial trace.",
    ),
    "qwen-local-wave-01": (
        ["pde_residual", "initial_condition", "boundary_condition"],
        "The added time exponential breaks the wave equation, both initial data, and the homogeneous spatial boundaries.",
    ),
    "qwen-local-laplace-01": (
        ["pde_residual", "boundary_condition"],
        "The y-only expression is not harmonic and does not match the prescribed left and right traces.",
    ),
    "qwen-local-poisson-01": (
        ["pde_residual", "boundary_condition"],
        "Direct differentiation does not recover the sine forcing, and the x=1 boundary is generally nonzero.",
    ),
    "qwen-local-transport-2d-01": (
        ["pde_residual"],
        "The candidate matches the initial profile but omits both characteristic shifts, leaving a nonzero transport residual.",
    ),
    "qwen-local-klein-gordon-01": (
        ["pde_residual", "initial_condition"],
        "The exponential time factor has the wrong second-time derivative and a nonzero initial velocity.",
    ),
}


def build() -> tuple[dict[str, object], dict[str, object]]:
    corpus = load_corpus("corpus/pilot.json")
    blank_records = []
    provisional_records = []
    for record in corpus["records"]:
        record_id = record["id"]
        blank_records.append(
            {"id": record_id, "verdict": None, "failure_modes": [], "rationale": None}
        )
        if record_id.startswith("sympy-"):
            proposal = {
                "id": record_id,
                "verdict": "valid",
                "failure_modes": [],
                "rationale": (
                    "Substitution gives an exact zero advection residual and the specialized "
                    "Fourier mode matches the stated initial and boundary traces."
                ),
            }
        else:
            modes, rationale = INVALID_PROPOSALS[record_id]
            proposal = {
                "id": record_id,
                "verdict": "invalid",
                "failure_modes": modes,
                "rationale": rationale,
            }
        provisional_records.append(proposal)
    return (
        {"review_version": 1, "records": blank_records},
        {"review_version": 1, "records": provisional_records},
    )


def main() -> None:
    blank, provisional = build()
    Path("corpus/review-template.json").write_text(
        json.dumps(blank, indent=2, sort_keys=True) + "\n"
    )
    Path("results/provisional-review.json").write_text(
        json.dumps(provisional, indent=2, sort_keys=True) + "\n"
    )
    print("Wrote corpus/review-template.json and results/provisional-review.json")


if __name__ == "__main__":
    main()
