"""Run one bounded, provider-backed PDE agent smoke case.

This is an integration and provenance check, not a model benchmark. The output
contains the exact public prompt and model response, but never provider tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pdecert import case_from_dict, case_to_dict, summarize_agent_runs
from pdecert.integrations.smolagents import run_smolagents_symbolic_agent


DEFAULT_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_PROVIDER = "novita"
PROMPT_VERSION = "heat-dirichlet-v1"
RESULT_SCHEMA_VERSION = 1


def heat_case():
    """Return the trusted strong-form case used by the smoke run."""

    return case_from_dict(
        {
            "schema_version": 3,
            "name": "one-dimensional heat equation",
            "variables": ["x", "t"],
            "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            "parameters": {},
            "pde_residuals": [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}],
            "conditions": [
                {"name": "initial condition", "expression": "At(u, t, 0) - sin(pi*x)"},
                {"name": "left boundary", "expression": "At(u, x, 0)"},
                {"name": "right boundary", "expression": "At(u, x, 1)"},
            ],
            "fields": {"u": "exp(-pi**2*t)*sin(pi*x)"},
        }
    )


def heat_prompt() -> str:
    """Return the exact public task shown to the model."""

    return (
        "Find u(x, t) on 0 <= x <= 1 and 0 <= t <= 1 satisfying "
        "D(u, t) - D(u, x, 2) = 0, u(x, 0) = sin(pi*x), and "
        "u(0, t) = u(1, t) = 0. Submit exactly one candidate field named u. "
        "Use Python/SymPy syntax."
    )


def resolve_live_revision(model_id: str, provider: str) -> str:
    """Return the Hub revision after confirming the requested provider is live."""

    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, expand=["inferenceProviderMapping", "sha"])
    mappings = info.inference_provider_mapping or []
    selected = next((item for item in mappings if item.provider == provider), None)
    if selected is None:
        available = ", ".join(sorted(item.provider for item in mappings)) or "none"
        raise RuntimeError(
            f"provider {provider!r} is unavailable; advertised providers: {available}"
        )
    if selected.status != "live":
        raise RuntimeError(f"provider {provider!r} is not live (status={selected.status!r})")
    if not isinstance(info.sha, str) or re.fullmatch(r"[0-9a-f]{40}", info.sha) is None:
        raise RuntimeError("Hugging Face did not return a full model revision")
    return info.sha


def repository_revision() -> str:
    """Return the exact PDECert source revision used for the run."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_result(
    *,
    run,
    model_id: str,
    model_revision: str,
    provider: str,
    max_steps: int,
    max_tokens: int,
    timeout_seconds: int,
    seed: int,
    pdecert_revision: str,
    generated_at: str,
    smolagents_version: str,
    huggingface_hub_version: str,
) -> dict[str, object]:
    """Build the deliberately public, self-describing smoke artifact."""

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite": "real-agent-smoke-v1",
        "scope": (
            "Single-case integration evidence; not independent ground-truth labeling or a "
            "population-level model comparison."
        ),
        "reproducibility_note": (
            "The Hub repository revision is recorded, but a hosted provider may update its "
            "serving stack or model deployment independently."
        ),
        "generated_at": generated_at,
        "prompt_version": PROMPT_VERSION,
        "prompt": heat_prompt(),
        "trusted_case": case_to_dict(heat_case()),
        "invocation": {
            "model_id": model_id,
            "model_revision": model_revision,
            "provider": provider,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed_requested": seed,
            "timeout_seconds": timeout_seconds,
        },
        "environment": {
            "pdecert_revision": pdecert_revision,
            "smolagents_version": smolagents_version,
            "huggingface_hub_version": huggingface_hub_version,
        },
        "run": run.to_dict(include_raw_outputs=True),
        "summary": summarize_agent_runs([run]).to_dict(),
    }


def run_experiment(
    *,
    run_id: str,
    model_id: str,
    provider: str,
    max_steps: int,
    max_tokens: int,
    timeout_seconds: int,
    seed: int,
) -> dict[str, object]:
    """Resolve provenance, execute one agent, and return its public artifact."""

    from smolagents import InferenceClientModel

    revision = resolve_live_revision(model_id, provider)
    source_revision = repository_revision()
    generator = f"{model_id}@{revision} via {provider}"
    model = InferenceClientModel(
        model_id=model_id,
        provider=provider,
        timeout=timeout_seconds,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    run = run_smolagents_symbolic_agent(
        trusted_case=heat_case(),
        model=model,
        prompt=heat_prompt(),
        run_id=run_id,
        problem_id="heat-dirichlet-01",
        generator=generator,
        max_steps=max_steps,
        metadata={
            "model_id": model_id,
            "model_revision": revision,
            "pdecert_revision": source_revision,
            "prompt_version": PROMPT_VERSION,
            "provider": provider,
            "seed_requested": str(seed),
        },
        agent_options={"verbosity_level": 0},
    )
    return build_result(
        run=run,
        model_id=model_id,
        model_revision=revision,
        provider=provider,
        max_steps=max_steps,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        seed=seed,
        pdecert_revision=source_revision,
        generated_at=datetime.now(timezone.utc).isoformat(),
        smolagents_version=version("smolagents"),
        huggingface_hub_version=version("huggingface_hub"),
    )


def write_new_json(path: Path, payload: object) -> None:
    """Atomically create a result without replacing earlier evidence."""

    if path.exists():
        raise FileExistsError(f"refusing to replace existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--max-steps", type=_positive, default=4)
    parser.add_argument("--max-tokens", type=_positive, default=768)
    parser.add_argument("--timeout-seconds", type=_positive, default=120)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    try:
        payload = run_experiment(
            run_id=arguments.run_id,
            model_id=arguments.model_id,
            provider=arguments.provider,
            max_steps=arguments.max_steps,
            max_tokens=arguments.max_tokens,
            timeout_seconds=arguments.timeout_seconds,
            seed=arguments.seed,
        )
        write_new_json(arguments.output, payload)
    except Exception as error:
        print(f"real_agent_smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"Wrote real-agent smoke result to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
