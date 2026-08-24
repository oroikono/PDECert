"""Run a predeclared open-model batch and materialize valid Atlas bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pdecert import case_from_dict, case_to_dict, output_sha256


_MANIFEST_FIELDS = {
    "batch_version",
    "cases",
    "description",
    "generation",
    "id",
    "model",
}
_MODEL_FIELDS = {"identifier", "license", "producer", "revision", "source_url"}
_CASE_FIELDS = {"fields", "id", "problem", "statement"}
_PROBLEM_FIELDS = {
    "conditions",
    "domains",
    "name",
    "parameters",
    "pde_residuals",
    "variables",
}
_RECORD_FILES = frozenset({"case.json", "raw-output.txt", "record.json"})


class CollectionError(ValueError):
    """Raised when a batch manifest or transcript is not reproducible."""


def _exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise CollectionError(f"{path}: missing field(s): {', '.join(missing)}")
    if unknown:
        raise CollectionError(f"{path}: unknown field(s): {', '.join(unknown)}")


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionError(f"{path}: expected an object")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionError(f"{path}: expected a non-empty string")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text()), str(path))
    except json.JSONDecodeError as error:
        raise CollectionError(f"{path}: invalid JSON: {error.msg}") from error


def _case_payload(case: dict[str, Any], fields: dict[str, str]) -> dict[str, object]:
    problem = case["problem"]
    payload = {
        "schema_version": 3,
        "name": problem["name"],
        "variables": problem["variables"],
        "domains": problem["domains"],
        "parameters": problem["parameters"],
        "fields": fields,
        "pde_residuals": problem["pde_residuals"],
        "conditions": problem["conditions"],
    }
    return case_to_dict(case_from_dict(payload))


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a version-one collection manifest."""

    source = Path(path)
    manifest = _read_json(source)
    _exact_keys(manifest, _MANIFEST_FIELDS, "$")
    if manifest["batch_version"] != 1 or isinstance(manifest["batch_version"], bool):
        raise CollectionError("$.batch_version: expected 1")
    _text(manifest["id"], "$.id")
    _text(manifest["description"], "$.description")

    model = _object(manifest["model"], "$.model")
    _exact_keys(model, _MODEL_FIELDS, "$.model")
    for name in sorted(_MODEL_FIELDS):
        _text(model[name], f"$.model.{name}")
    revision = model["revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise CollectionError("$.model.revision: expected a full 40-character Git revision")

    generation = _object(manifest["generation"], "$.generation")
    _exact_keys(generation, {"do_sample", "max_new_tokens"}, "$.generation")
    if generation["do_sample"] is not False:
        raise CollectionError("$.generation.do_sample: deterministic batches require false")
    tokens = generation["max_new_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise CollectionError("$.generation.max_new_tokens: expected a positive integer")

    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise CollectionError("$.cases: expected a non-empty list")
    identifiers: set[str] = set()
    for index, raw_case in enumerate(cases):
        path_prefix = f"$.cases[{index}]"
        case = _object(raw_case, path_prefix)
        _exact_keys(case, _CASE_FIELDS, path_prefix)
        record_id = _text(case["id"], f"{path_prefix}.id")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", record_id) is None:
            raise CollectionError(f"{path_prefix}.id: expected a lowercase record identifier")
        if record_id in identifiers:
            raise CollectionError(f"{path_prefix}.id: duplicate record identifier")
        identifiers.add(record_id)
        _text(case["statement"], f"{path_prefix}.statement")
        fields = case["fields"]
        if not isinstance(fields, list) or not fields or any(
            not isinstance(field, str) or not field.isidentifier() for field in fields
        ):
            raise CollectionError(f"{path_prefix}.fields: expected Python identifiers")
        if len(set(fields)) != len(fields):
            raise CollectionError(f"{path_prefix}.fields: names must be unique")
        problem = _object(case["problem"], f"{path_prefix}.problem")
        _exact_keys(problem, _PROBLEM_FIELDS, f"{path_prefix}.problem")
        _case_payload(case, {field: "0" for field in fields})
    return manifest


def prompt_for(case: dict[str, Any]) -> str:
    """Build the exact model prompt for one predeclared case."""

    final_lines = "\n".join(f"FINAL {field}: <expression>" for field in case["fields"])
    variables = ", ".join(case["problem"]["variables"])
    return (
        f"{case['statement']}\n\n"
        "Return one explicit candidate expression for every requested field. "
        f"Use Python/SymPy syntax with variables {variables}. You may use pi, E, "
        "sin, cos, sinh, cosh, tanh, exp, log, sqrt, Abs, +, -, *, /, **, and "
        "parentheses. Do not return an implicit equation, arbitrary function, "
        "derivative, integral, code block, or explanation. End with exactly:\n"
        f"{final_lines}"
    )


def _normalize_expression(source: str) -> str:
    expression = source.strip().strip("`").strip()
    assignment = re.match(r"^[A-Za-z_]\w*(?:\([^=]+\))?\s*=\s*(.+)$", expression)
    if assignment:
        expression = assignment.group(1).strip()
    return expression.replace("^", "**").replace("π", "pi")


def extract_fields(response: str, expected: list[str]) -> dict[str, str]:
    """Extract exactly one final expression for every expected field."""

    named = re.findall(
        r"^FINAL\s+([A-Za-z_]\w*)\s*:\s*(.+?)\s*$",
        response,
        flags=re.MULTILINE,
    )
    if len(expected) == 1 and not named:
        unnamed = re.findall(r"^FINAL\s*:\s*(.+?)\s*$", response, flags=re.MULTILINE)
        if len(unnamed) == 1:
            return {expected[0]: _normalize_expression(unnamed[0])}
    extracted: dict[str, str] = {}
    for name, expression in named:
        if name not in expected:
            raise CollectionError(f"model returned unknown field: {name}")
        if name in extracted:
            raise CollectionError(f"model returned field more than once: {name}")
        extracted[name] = _normalize_expression(expression)
    missing = [name for name in expected if name not in extracted]
    if missing:
        raise CollectionError(f"model did not return field(s): {', '.join(missing)}")
    return {name: extracted[name] for name in expected}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_observed_model(
    manifest: dict[str, Any], *, local_files_only: bool = False
) -> None:
    from huggingface_hub import HfApi, snapshot_download

    model = manifest["model"]
    if local_files_only:
        snapshot = Path(
            snapshot_download(
                repo_id=model["identifier"],
                revision=model["revision"],
                local_files_only=True,
            )
        ).resolve()
        observed = snapshot.name
    else:
        observed = HfApi().model_info(model["identifier"], revision=model["revision"]).sha
    if observed != model["revision"]:
        raise CollectionError(
            f"model revision mismatch: expected {model['revision']}, observed {observed}"
        )


def generate(
    manifest: dict[str, Any],
    run_directory: Path,
    *,
    local_files_only: bool = False,
) -> None:
    """Generate every missing transcript once with the pinned open model."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install requirements/atlas-collection.txt") from error

    _validate_observed_model(manifest, local_files_only=local_files_only)
    model_info = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["identifier"],
        revision=model_info["revision"],
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_info["identifier"],
        revision=model_info["revision"],
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    run_directory.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        destination = run_directory / f"{case['id']}.json"
        if destination.exists():
            print(f"skip {case['id']}: transcript exists")
            continue
        prompt = prompt_for(case)
        messages = [
            {
                "role": "system",
                "content": "Solve the stated PDE problem and follow the output grammar exactly.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=manifest["generation"]["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        transcript = {
            "batch_id": manifest["id"],
            "case_id": case["id"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation": manifest["generation"],
            "model": model_info,
            "prompt": prompt,
            "response": response,
            "transformers_version": transformers.__version__,
        }
        _atomic_json(destination, transcript)
        print(f"saved {case['id']}")


def _pending_annotation() -> dict[str, object]:
    return {
        "annotators": [],
        "failure_modes": [],
        "rationale": None,
        "status": "pending",
        "verdict": None,
    }


def _bundle_payloads(
    manifest: dict[str, Any],
    case: dict[str, Any],
    transcript: dict[str, Any],
) -> dict[str, bytes]:
    if transcript.get("batch_id") != manifest["id"] or transcript.get("case_id") != case["id"]:
        raise CollectionError("transcript batch or case identifier does not match")
    if transcript.get("model") != manifest["model"]:
        raise CollectionError("transcript model metadata does not match the manifest")
    if transcript.get("generation") != manifest["generation"]:
        raise CollectionError("transcript generation settings do not match the manifest")
    if transcript.get("prompt") != prompt_for(case):
        raise CollectionError("transcript prompt does not match the predeclared case")
    raw_output = _text(transcript.get("response"), "$.response")
    fields = extract_fields(raw_output, case["fields"])
    serialized_case = _case_payload(case, fields)
    origin_input = json.dumps(
        {"generation": manifest["generation"], "prompt": transcript["prompt"]},
        sort_keys=True,
    )
    metadata = {
        "annotation": _pending_annotation(),
        "id": case["id"],
        "origin": {
            "generated_at": transcript["generated_at"],
            "identifier": manifest["model"]["identifier"],
            "input": origin_input,
            "kind": "open_model",
            "license": manifest["model"]["license"],
            "producer": manifest["model"]["producer"],
            "revision": manifest["model"]["revision"],
            "source_url": manifest["model"]["source_url"],
            "version": transcript["transformers_version"],
        },
        "output_sha256": output_sha256(raw_output),
    }
    return {
        "case.json": (json.dumps(serialized_case, indent=2, sort_keys=True) + "\n").encode(),
        "raw-output.txt": raw_output.encode(),
        "record.json": (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    }


def _write_bundle(directory: Path, payloads: dict[str, bytes]) -> None:
    if directory.exists():
        existing = {path.name for path in directory.iterdir()}
        if existing != _RECORD_FILES:
            raise CollectionError(f"{directory}: existing bundle has unexpected files")
        for name, content in payloads.items():
            if (directory / name).read_bytes() != content:
                raise CollectionError(f"{directory}: refusing to overwrite different {name}")
        return
    directory.mkdir(parents=True)
    for name, content in payloads.items():
        (directory / name).write_bytes(content)


def materialize(
    manifest: dict[str, Any],
    run_directory: Path,
    atlas_directory: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Materialize every parseable transcript and report every attempted case."""

    outcomes: list[dict[str, str]] = []
    for case in manifest["cases"]:
        transcript_path = run_directory / f"{case['id']}.json"
        if not transcript_path.exists():
            outcomes.append({"id": case["id"], "status": "missing_transcript"})
            continue
        transcript = _read_json(transcript_path)
        response = transcript.get("response")
        digest = hashlib.sha256(response.encode()).hexdigest() if isinstance(response, str) else ""
        try:
            payloads = _bundle_payloads(manifest, case, transcript)
            _write_bundle(atlas_directory / "records" / case["id"], payloads)
        except (CollectionError, ValueError) as error:
            outcomes.append(
                {
                    "error": str(error),
                    "id": case["id"],
                    "raw_output_sha256": digest,
                    "status": "not_materialized",
                }
            )
            continue
        outcomes.append(
            {
                "id": case["id"],
                "raw_output_sha256": digest,
                "status": "materialized_pending_review",
            }
        )
    report = {
        "batch_id": manifest["id"],
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "model": manifest["model"],
        "outcomes": outcomes,
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "materialize"):
        command = subcommands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--run-directory", type=Path, required=True)
        if name == "generate":
            command.add_argument(
                "--local-files-only",
                action="store_true",
                help="resolve and load the pinned revision from the local Hugging Face cache",
            )
        if name == "materialize":
            command.add_argument("--atlas", type=Path, default=Path("corpus/community"))
            command.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = load_manifest(arguments.manifest)
    if arguments.command == "generate":
        generate(
            manifest,
            arguments.run_directory,
            local_files_only=arguments.local_files_only,
        )
    else:
        report = materialize(
            manifest,
            arguments.run_directory,
            arguments.atlas,
            arguments.report,
        )
        counts: dict[str, int] = {}
        for outcome in report["outcomes"]:
            counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
        print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
