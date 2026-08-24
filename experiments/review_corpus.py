"""Run a blind, resumable review of candidate-corpus records."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pdecert import FAILURE_MODES, CorpusError, load_corpus_source


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class ReviewSessionError(ValueError):
    """Raised when a saved review cannot be resumed safely."""


def new_review(corpus: Mapping[str, object]) -> dict[str, object]:
    """Return an empty review document with the corpus record order."""

    return {
        "review_version": 1,
        "records": [
            {
                "failure_modes": [],
                "id": record["id"],
                "rationale": None,
                "verdict": None,
            }
            for record in corpus["records"]
        ],
    }


def _validate_resume(review: object, corpus: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(review, dict) or set(review) != {"records", "review_version"}:
        raise ReviewSessionError("saved review has an unsupported structure")
    if (
        isinstance(review["review_version"], bool)
        or review["review_version"] != 1
        or not isinstance(review["records"], list)
    ):
        raise ReviewSessionError("saved review has an unsupported version")
    expected_ids = [record["id"] for record in corpus["records"]]
    actual_ids = [record.get("id") for record in review["records"] if isinstance(record, dict)]
    if actual_ids != expected_ids or len(actual_ids) != len(review["records"]):
        raise ReviewSessionError("saved review IDs do not match the corpus in order")
    required = {"failure_modes", "id", "rationale", "verdict"}
    for record in review["records"]:
        if set(record) != required:
            raise ReviewSessionError(f"saved decision has unsupported fields: {record['id']}")
        verdict = record["verdict"]
        if verdict is None:
            if record["failure_modes"] or record["rationale"] is not None:
                raise ReviewSessionError(f"incomplete saved decision: {record['id']}")
            continue
        if verdict not in {"valid", "invalid", "unclear"}:
            raise ReviewSessionError(f"unsupported verdict in saved review: {record['id']}")
        if not isinstance(record["rationale"], str) or not record["rationale"].strip():
            raise ReviewSessionError(f"completed decision lacks a rationale: {record['id']}")
        modes = record["failure_modes"]
        if (
            not isinstance(modes, list)
            or any(not isinstance(mode, str) for mode in modes)
            or set(modes) - FAILURE_MODES
            or len(set(modes)) != len(modes)
        ):
            raise ReviewSessionError(f"unsupported failure mode in saved review: {record['id']}")
        if verdict == "invalid" and not modes:
            raise ReviewSessionError(f"invalid decision lacks a failure mode: {record['id']}")
        if verdict != "invalid" and modes:
            raise ReviewSessionError(f"non-invalid decision has failure modes: {record['id']}")
    return review


def load_or_create_review(path: Path, corpus: Mapping[str, object]) -> dict[str, object]:
    """Load a compatible saved review or create a new in-memory review."""

    if not path.exists():
        return new_review(corpus)
    try:
        review = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ReviewSessionError(f"saved review is not valid JSON: {error.msg}") from error
    return _validate_resume(review, corpus)


def save_review(path: Path, review: Mapping[str, object]) -> None:
    """Save after one decision so a session can be resumed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def progress_bar(completed: int, total: int, width: int = 20) -> str:
    filled = round(width * completed / total) if total else width
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total}"


def render_record(record: Mapping[str, object], position: int, total: int) -> str:
    """Render only source material allowed during the blind pass."""

    case = record["case"]
    fields = "\n".join(f"  {name} = {expression}" for name, expression in case["fields"].items())
    residuals = "\n".join(
        f"  - {constraint['name']}: {constraint['expression']}"
        for constraint in case["pde_residuals"]
    )
    conditions = (
        "\n".join(
            f"  - {constraint['name']}: {constraint['expression']}"
            for constraint in case["conditions"]
        )
        or "  (none)"
    )
    return (
        f"\n{'=' * 72}\n"
        f"CARD {position}/{total}: {record['id']}\n"
        f"Problem: {case['name']}\n"
        f"Variables: {', '.join(case['variables'])}\n"
        f"Domains: {json.dumps(case['domains'], sort_keys=True)}\n\n"
        f"Candidate fields:\n{fields}\n\n"
        f"PDE residuals:\n{residuals}\n\n"
        f"Conditions:\n{conditions}\n\n"
        f"Unedited generator output:\n{record['raw_output']}\n"
    )


def _prompt_verdict(input_fn: InputFunction, output_fn: OutputFunction) -> str:
    while True:
        answer = (
            input_fn("Verdict [v=valid, i=invalid, u=unclear, s=skip, q=quit]: ").strip().lower()
        )
        verdict = {"v": "valid", "i": "invalid", "u": "unclear"}.get(answer)
        if verdict is not None:
            return verdict
        if answer in {"s", "q"}:
            return answer
        output_fn("Please enter v, i, u, s, or q.")


def _prompt_failure_modes(input_fn: InputFunction, output_fn: OutputFunction) -> list[str]:
    modes = sorted(FAILURE_MODES)
    output_fn("Failure modes:")
    for index, mode in enumerate(modes, start=1):
        output_fn(f"  {index}. {mode}")
    while True:
        answer = input_fn("Select one or more numbers, separated by commas: ").strip()
        try:
            indices = [int(item.strip()) for item in answer.split(",") if item.strip()]
        except ValueError:
            indices = []
        if (
            indices
            and len(indices) == len(set(indices))
            and all(1 <= item <= len(modes) for item in indices)
        ):
            return [modes[item - 1] for item in indices]
        output_fn("Choose one or more unique numbers from the list.")


def _prompt_rationale(input_fn: InputFunction, output_fn: OutputFunction) -> str:
    while True:
        rationale = input_fn("Short mathematical rationale: ").strip()
        if rationale:
            return rationale
        output_fn("A non-empty rationale is required.")


def run_session(
    corpus: Mapping[str, object],
    review: dict[str, object],
    output_path: Path,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> bool:
    """Review pending cards and return whether all cards are complete."""

    decisions = {record["id"]: record for record in review["records"]}
    total = len(corpus["records"])
    completed = sum(record["verdict"] is not None for record in review["records"])
    output_fn(f"Blind review progress {progress_bar(completed, total)}")
    output_fn("No verifier result or provisional suggestion is shown.")

    for position, record in enumerate(corpus["records"], start=1):
        decision = decisions[record["id"]]
        if decision["verdict"] is not None:
            continue
        output_fn(render_record(record, position, total))
        try:
            verdict = _prompt_verdict(input_fn, output_fn)
            failure_modes = (
                _prompt_failure_modes(input_fn, output_fn) if verdict == "invalid" else []
            )
            rationale = (
                _prompt_rationale(input_fn, output_fn) if verdict not in {"q", "s"} else None
            )
        except (EOFError, KeyboardInterrupt):
            save_review(output_path, review)
            output_fn(f"\nSaved. Resume with the same command. {progress_bar(completed, total)}")
            return False
        if verdict == "q":
            save_review(output_path, review)
            output_fn(f"Saved. Resume with the same command. {progress_bar(completed, total)}")
            return False
        if verdict == "s":
            continue
        decision.update(
            verdict=verdict,
            failure_modes=failure_modes,
            rationale=rationale,
        )
        completed += 1
        save_review(output_path, review)
        output_fn(f"Saved {record['id']}. {progress_bar(completed, total)}")

    complete = completed == total
    if complete:
        output_fn(f"Blind pass complete. {progress_bar(completed, total)}")
        output_fn("Now follow corpus/LABELING.md for comparison and disagreement checks.")
    else:
        output_fn(f"Skipped cards remain. Resume later. {progress_bar(completed, total)}")
    return complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus/pilot.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("private-reviews/pilot-review.json"),
        help="private resumable review file",
    )
    arguments = parser.parse_args(argv)
    try:
        corpus = load_corpus_source(arguments.corpus)
        review = load_or_create_review(arguments.output, corpus)
        run_session(corpus, review, arguments.output)
    except (CorpusError, OSError, ReviewSessionError) as error:
        print(f"review_corpus: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
