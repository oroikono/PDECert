"""Collect the 20-record pilot from live symbolic-solver and open-model runs."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sympy as sp

from pdecert import (
    SchemaError,
    case_from_dict,
    case_to_dict,
    dump_corpus,
    output_sha256,
    validate_corpus,
    verify,
)


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_PROVIDER = "featherless-ai"
MODEL_SOURCE = f"https://huggingface.co/{MODEL_ID}"
LOCAL_MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
LOCAL_MODEL_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
LOCAL_MODEL_SOURCE = f"https://huggingface.co/{LOCAL_MODEL_ID}"
SYMPY_SOURCE = "https://docs.sympy.org/latest/modules/solvers/pde.html"
RAW_OUTPUT_DIRECTORY = Path(__file__).parents[1] / "corpus" / "raw"


@dataclass(frozen=True)
class ModelProblem:
    record_id: str
    name: str
    variables: tuple[str, ...]
    domains: dict[str, list[float]]
    pde_residuals: tuple[tuple[str, str], ...]
    conditions: tuple[tuple[str, str], ...]
    statement: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pending_annotation() -> dict[str, object]:
    return {
        "status": "pending",
        "verdict": None,
        "failure_modes": [],
        "rationale": None,
        "annotators": [],
    }


def _case_payload(problem: ModelProblem, expression: str) -> dict[str, object]:
    payload = {
        "schema_version": 3,
        "name": problem.name,
        "variables": list(problem.variables),
        "domains": problem.domains,
        "parameters": {},
        "fields": {"u": expression},
        "pde_residuals": [
            {"name": name, "expression": residual} for name, residual in problem.pde_residuals
        ],
        "conditions": [
            {"name": name, "expression": residual} for name, residual in problem.conditions
        ],
    }
    return case_to_dict(case_from_dict(payload))


def _record(
    *,
    record_id: str,
    case: dict[str, object],
    raw_output: str,
    origin: dict[str, object],
) -> dict[str, object]:
    return {
        "id": record_id,
        "case": case,
        "origin": origin,
        "raw_output": raw_output,
        "output_sha256": output_sha256(raw_output),
        "annotation": _pending_annotation(),
    }


def _save_raw_output(record_id: str, raw_output: str) -> None:
    RAW_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (RAW_OUTPUT_DIRECTORY / f"{record_id}.txt").write_text(raw_output)


def collect_sympy_records() -> list[dict[str, object]]:
    """Run ten real pdsolve calls and specialize their returned characteristic functions."""

    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    records: list[dict[str, object]] = []
    settings = ((1, 1), (1, 2), (2, 1), (2, 3), (3, 1), (3, 2), (-1, 1), (-2, 2), (-3, 3), (4, 2))
    for index, (speed, mode) in enumerate(settings, start=1):
        equation = sp.Eq(sp.diff(u(x, t), t) + speed * sp.diff(u(x, t), x), 0)
        solution = sp.pdsolve(equation, u(x, t))
        raw_output = sp.sstr(solution)
        candidate = sp.sin(mode * sp.pi * (x - speed * t))
        expression = sp.sstr(candidate)
        problem = ModelProblem(
            record_id=f"sympy-advection-{index:02d}",
            name=f"advection speed {speed}, Fourier mode {mode}",
            variables=("x", "t"),
            domains={"x": [0.0, 1.0], "t": [0.0, 1.0]},
            pde_residuals=(("advection PDE", f"D(u, t) + ({speed})*D(u, x)"),),
            conditions=(
                ("initial condition", f"At(u, t, 0) - sin(({mode})*pi*x)"),
                (
                    "left boundary",
                    f"At(u, x, 0) - sin(({mode})*pi*(-({speed})*t))",
                ),
                (
                    "right boundary",
                    f"At(u, x, 1) - sin(({mode})*pi*(1-({speed})*t))",
                ),
            ),
            statement="",
        )
        case = _case_payload(problem, expression)
        loaded = case_from_dict(case)
        if verify(loaded.problem, loaded.candidate_fields).status.value != "PROVED":
            raise RuntimeError(f"specialized SymPy candidate did not verify: {problem.record_id}")
        exact_input = (
            f"pdsolve({sp.sstr(equation)}, u(x, t)); returned {raw_output}; "
            f"specialize its arbitrary F using u(x,0)=sin({mode}*pi*x), yielding {expression}"
        )
        records.append(
            _record(
                record_id=problem.record_id,
                case=case,
                raw_output=raw_output,
                origin={
                    "kind": "symbolic_solver",
                    "producer": "SymPy",
                    "version": sp.__version__,
                    "identifier": "sympy.solvers.pde.pdsolve",
                    "revision": None,
                    "source_url": SYMPY_SOURCE,
                    "license": "BSD-3-Clause",
                    "generated_at": _timestamp(),
                    "input": exact_input,
                },
            )
        )
    return records


def _model_problems() -> tuple[ModelProblem, ...]:
    return (
        ModelProblem(
            "qwen-heat-mode-01",
            "heat equation, first mode",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("heat PDE", "D(u, t) - D(u, x, 2)"),),
            (
                ("initial", "At(u, t, 0) - sin(pi*x)"),
                ("left", "At(u, x, 0)"),
                ("right", "At(u, x, 1)"),
            ),
            "Solve u_t = u_xx with u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0.",
        ),
        ModelProblem(
            "qwen-heat-mode-02",
            "heat equation, second mode",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("heat PDE", "D(u, t) - D(u, x, 2)"),),
            (
                ("initial", "At(u, t, 0) - sin(2*pi*x)"),
                ("left", "At(u, x, 0)"),
                ("right", "At(u, x, 1)"),
            ),
            "Solve u_t = u_xx with u(x,0)=sin(2*pi*x), u(0,t)=u(1,t)=0.",
        ),
        ModelProblem(
            "qwen-transport-01",
            "unit-speed transport",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("transport PDE", "D(u, t) + D(u, x)"),),
            (("initial", "At(u, t, 0) - sin(pi*x)"),),
            "Solve u_t + u_x = 0 with u(x,0)=sin(pi*x).",
        ),
        ModelProblem(
            "qwen-transport-02",
            "speed-two transport",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("transport PDE", "D(u, t) + 2*D(u, x)"),),
            (("initial", "At(u, t, 0) - cos(pi*x)"),),
            "Solve u_t + 2*u_x = 0 with u(x,0)=cos(pi*x).",
        ),
        ModelProblem(
            "qwen-reaction-01",
            "linear reaction equation",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("reaction PDE", "D(u, t) + u"),),
            (("initial", "At(u, t, 0) - sin(pi*x)"),),
            "Solve u_t + u = 0 with u(x,0)=sin(pi*x).",
        ),
        ModelProblem(
            "qwen-wave-01",
            "wave equation, first mode",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("wave PDE", "D(u, t, 2) - D(u, x, 2)"),),
            (
                ("position", "At(u, t, 0) - sin(pi*x)"),
                ("velocity", "At(D(u, t), t, 0)"),
                ("left", "At(u, x, 0)"),
                ("right", "At(u, x, 1)"),
            ),
            "Solve u_tt = u_xx with u(x,0)=sin(pi*x), u_t(x,0)=0, u(0,t)=u(1,t)=0.",
        ),
        ModelProblem(
            "qwen-laplace-01",
            "Laplace equation with exponential trace",
            ("x", "y"),
            {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            (("Laplace PDE", "D(u, x, 2) + D(u, y, 2)"),),
            (
                ("bottom", "At(u, y, 0)"),
                ("top", "At(u, y, 1)"),
                ("left", "At(u, x, 0) - sin(pi*y)"),
                ("right", "At(u, x, 1) - exp(pi)*sin(pi*y)"),
            ),
            "Solve u_xx + u_yy = 0 on [0,1]^2 with u(x,0)=u(x,1)=0, u(0,y)=sin(pi*y), u(1,y)=exp(pi)*sin(pi*y).",
        ),
        ModelProblem(
            "qwen-poisson-01",
            "Poisson equation, sine forcing",
            ("x", "y"),
            {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            (("Poisson PDE", "D(u, x, 2) + D(u, y, 2) + 2*pi**2*sin(pi*x)*sin(pi*y)"),),
            (
                ("x zero", "At(u, x, 0)"),
                ("x one", "At(u, x, 1)"),
                ("y zero", "At(u, y, 0)"),
                ("y one", "At(u, y, 1)"),
            ),
            "Solve u_xx + u_yy = -2*pi^2*sin(pi*x)*sin(pi*y) on [0,1]^2 with zero boundary values.",
        ),
        ModelProblem(
            "qwen-transport-2d-01",
            "two-dimensional transport",
            ("x", "y", "t"),
            {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
            (("transport PDE", "D(u, t) + D(u, x) + D(u, y)"),),
            (("initial", "At(u, t, 0) - sin(pi*x)*sin(pi*y)"),),
            "Solve u_t + u_x + u_y = 0 with u(x,y,0)=sin(pi*x)*sin(pi*y).",
        ),
        ModelProblem(
            "qwen-klein-gordon-01",
            "linear Klein-Gordon equation",
            ("x", "t"),
            {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            (("Klein-Gordon PDE", "D(u, t, 2) - D(u, x, 2) + u"),),
            (
                ("position", "At(u, t, 0) - sin(pi*x)"),
                ("velocity", "At(D(u, t), t, 0)"),
                ("left", "At(u, x, 0)"),
                ("right", "At(u, x, 1)"),
            ),
            "Solve u_tt - u_xx + u = 0 with u(x,0)=sin(pi*x), u_t(x,0)=0, u(0,t)=u(1,t)=0.",
        ),
    )


def _prompt(problem: ModelProblem) -> str:
    return (
        f"{problem.statement}\n"
        "Return one explicit candidate expression for u in Python/SymPy syntax. "
        "Allowed names are the stated variables, pi, E, sin, cos, exp, sqrt, +, -, *, /, **, and parentheses. "
        "Answer with one line beginning FINAL: and no prose."
    )


def _extract_expression(content: str) -> str:
    matches = re.findall(r"^FINAL:\s*(.+?)\s*$", content, flags=re.MULTILINE)
    if matches:
        expression = matches[-1]
    else:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError("model output does not contain one extractable expression")
        expression = lines[0]
    expression = expression.strip().strip("`")
    assignment = re.match(r"^u(?:\([^=]+\))?\s*=\s*(.+)$", expression)
    if assignment:
        expression = assignment.group(1).strip()
    expression = expression.replace("^", "**")
    expression = expression.replace("π", "pi")
    expression = re.sub(r"(?<=[0-9)])(?=pi)", "*", expression)
    expression = re.sub(r"(?<=pi)(?=[xyt(])", "*", expression)
    expression = re.sub(r"(?<=[0-9)])(?=[xyt])", "*", expression)
    return expression


def collect_open_model_records() -> list[dict[str, object]]:
    """Request ten candidates from the pinned open model through one named provider."""

    try:
        import huggingface_hub
        from huggingface_hub import HfApi, InferenceClient, get_token
    except ImportError as error:
        raise RuntimeError("install the collection extra before open-model collection") from error

    token = get_token()
    if token is None:
        raise RuntimeError("authenticate with hf auth login before open-model collection")
    observed_revision = HfApi(token=token).model_info(MODEL_ID).sha
    if observed_revision != MODEL_REVISION:
        raise RuntimeError(
            f"model revision changed: expected {MODEL_REVISION}, observed {observed_revision}"
        )
    client = InferenceClient(provider=MODEL_PROVIDER, token=token, timeout=120)
    records: list[dict[str, object]] = []
    for index, problem in enumerate(_model_problems(), start=1):
        prompt = _prompt(problem)
        completion = client.chat_completion(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "Follow the requested output grammar exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=96,
            temperature=0.6,
            top_p=0.95,
            presence_penalty=1.5,
            seed=20_260_823 + index,
        )
        content = completion.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError(f"model returned no text for {problem.record_id}")
        expression = _extract_expression(content)
        case = _case_payload(problem, expression)
        response_payload = (
            completion.model_dump() if hasattr(completion, "model_dump") else completion
        )
        raw_output = json.dumps(response_payload, ensure_ascii=False, sort_keys=True, default=str)
        _save_raw_output(problem.record_id, raw_output)
        records.append(
            _record(
                record_id=problem.record_id,
                case=case,
                raw_output=raw_output,
                origin={
                    "kind": "open_model",
                    "producer": f"Qwen via {MODEL_PROVIDER}",
                    "version": huggingface_hub.__version__,
                    "identifier": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "source_url": MODEL_SOURCE,
                    "license": "Apache-2.0",
                    "generated_at": _timestamp(),
                    "input": prompt,
                },
            )
        )
    return records


def collect_local_model_records() -> list[dict[str, object]]:
    """Generate ten candidates locally with a pinned 4-bit MLX model."""

    try:
        import mlx.core as mx
        import mlx_lm
        from huggingface_hub import snapshot_download
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as error:
        raise RuntimeError("install the collection extra on Apple silicon") from error

    model_path = snapshot_download(LOCAL_MODEL_ID, revision=LOCAL_MODEL_REVISION)
    model, tokenizer = load(model_path)
    sampler = make_sampler(temp=0.2, top_p=0.9, min_p=0.0)
    records: list[dict[str, object]] = []
    for index, problem in enumerate(_model_problems(), start=1):
        prompt = _prompt(problem)
        messages = [
            {"role": "system", "content": "Return only a SymPy expression after FINAL:."},
            {"role": "user", "content": prompt},
        ]
        record_id = problem.record_id.replace("qwen-", "qwen-local-")
        responses: list[str] = []
        for attempt in range(3):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            mx.random.seed(20_260_823 + index + attempt * 1_000)
            content = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=64,
                sampler=sampler,
                verbose=False,
            )
            responses.append(content)
            raw_output = json.dumps({"responses": responses}, ensure_ascii=False, sort_keys=True)
            _save_raw_output(record_id, raw_output)
            try:
                expression = _extract_expression(content)
                case = _case_payload(problem, expression)
                break
            except (SchemaError, ValueError):
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "That is not a parseable explicit expression for u. "
                                "Do not write a differential equation and do not use u on the right. "
                                "Return one concrete formula, not a placeholder. "
                                "Example format: FINAL: exp(-t)*sin(pi*x)."
                            ),
                        },
                    ]
                )
        else:
            raise RuntimeError(f"no extractable candidate after three attempts: {record_id}")
        records.append(
            _record(
                record_id=record_id,
                case=case,
                raw_output=raw_output,
                origin={
                    "kind": "open_model",
                    "producer": "Qwen via MLX LM",
                    "version": mlx_lm.__version__,
                    "identifier": LOCAL_MODEL_ID,
                    "revision": LOCAL_MODEL_REVISION,
                    "source_url": LOCAL_MODEL_SOURCE,
                    "license": "Apache-2.0",
                    "generated_at": _timestamp(),
                    "input": json.dumps(
                        {
                            "initial_prompt": prompt,
                            "maximum_attempts": 3,
                            "repair_prompt": (
                                "That is not a parseable explicit expression for u. "
                                "Do not write a differential equation and do not use u on the right. "
                                "Return one concrete formula, not a placeholder. "
                                "Example format: FINAL: exp(-t)*sin(pi*x)."
                            ),
                        },
                        sort_keys=True,
                    ),
                },
            )
        )
    return records


def collect(*, backend: str = "local") -> dict[str, object]:
    if backend not in {"local", "provider"}:
        raise ValueError("backend must be local or provider")
    model_records = (
        collect_local_model_records() if backend == "local" else collect_open_model_records()
    )
    records = collect_sympy_records() + model_records
    if len(records) != 20:
        raise RuntimeError(f"expected 20 records, collected {len(records)}")
    corpus = {
        "corpus_version": 1,
        "name": "PDECert natural-candidate pilot",
        "description": "Unedited outputs from ten SymPy pdsolve runs and ten local open-model generations.",
        "records": records,
    }
    validate_corpus(corpus)
    return corpus


def main() -> None:
    backend = "provider" if "--provider" in sys.argv[1:] else "local"
    output = Path(__file__).parents[1] / "corpus" / "pilot.json"
    dump_corpus(collect(backend=backend), output)
    print(f"Wrote 20 records to {output}")


if __name__ == "__main__":
    main()
