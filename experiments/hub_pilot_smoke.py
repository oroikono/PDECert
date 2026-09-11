"""Check the frozen pilot through installed Hugging Face Datasets, without inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from pdecert import (
    build_release_bundle,
    case_from_dict,
    corpus_sha256,
    load_corpus,
    validate_corpus,
    verify,
)


PILOT_REPO = "oroikono/pdecert-pilot"
PILOT_REVISION = "db690f9b161762ea288dd5dfb4b6b2f999c48e03"
PILOT_SHA256 = "4be9178edd30fcc561f21e83375713f4b38338484d75a0c8a7c8088e9c4369fb"
CONFIG = "default"
SPLIT = "test"
DATA_FILE = "data/pilot.jsonl"
ROOT = Path(__file__).resolve().parents[1]
VERIFY_OPTIONS = {
    "tolerance": 1e-9,
    "samples_per_axis": 5,
    "symbolic_timeout": 2.0,
    "max_expression_ops": 10_000,
}


class HubRoundtripError(ValueError):
    """A transport check failed; this is not a PDE verdict."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def _offline_loading():
    """Disable local-loader requests, restoring the pinned libraries' settings.

    The standalone smoke runner is single-threaded. These process-global library
    switches are not a security sandbox or a concurrent service interface.
    """

    from datasets import config
    from huggingface_hub import constants

    previous = config.HF_HUB_OFFLINE, constants.HF_HUB_OFFLINE
    try:
        config.HF_HUB_OFFLINE = constants.HF_HUB_OFFLINE = True
        yield
    finally:
        config.HF_HUB_OFFLINE, constants.HF_HUB_OFFLINE = previous


def check_loaded_rows(corpus: dict, rows: list[dict]) -> dict:
    """Require exact JSON fidelity; do not repair, strip nulls, or reparse fields.

    This is a reference-guided release check, not a general Hub record importer.
    A future loader that changes the nested representation must fail this check
    until its compatibility has been reviewed explicitly.
    """

    validate_corpus(corpus)
    expected_rows = corpus["records"]
    if len(rows) != len(expected_rows):
        raise HubRoundtripError("loaded row count differs from the source corpus")
    restored = copy.deepcopy(rows)
    for index, (expected, actual) in enumerate(zip(expected_rows, restored)):
        # JSON equality detects bool/int and int/float changes that Python == hides.
        if _canonical(actual) != _canonical(expected):
            raise HubRoundtripError(f"row {index}: loaded content differs from the source")
    rebuilt = {**corpus, "records": restored}
    validate_corpus(rebuilt)
    if corpus_sha256(rebuilt) != corpus_sha256(corpus):
        raise HubRoundtripError("restored corpus digest differs from the source")
    return rebuilt


def evaluate_loaded_rows(corpus: dict) -> list[dict]:
    """Run the ordinary verifier; never derive decisions from the stored labels."""

    validate_corpus(corpus)
    results = []
    for row in corpus["records"]:
        case = case_from_dict(row["case"])
        report = verify(case.problem, case.candidate_fields, **VERIFY_OPTIONS)
        results.append({"id": row["id"], "report": report.to_dict()})
    return results


def run_smoke(*, from_hub: bool = False) -> dict:
    """Check real JSON-to-Arrow loading locally, or anonymously at a fixed Hub SHA."""

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise HubRoundtripError(
            "install the optional requirements/hub-integration.txt dependencies"
        ) from error

    corpus = load_corpus(ROOT / "corpus/pilot.json")
    if corpus_sha256(corpus) != PILOT_SHA256:
        raise HubRoundtripError("source corpus no longer matches the frozen pilot digest")
    benchmark = json.loads((ROOT / "results/pilot-benchmark.json").read_text())
    with tempfile.TemporaryDirectory(prefix="pdecert-hub-smoke-") as directory:
        scratch = Path(directory)
        bundle = scratch / "bundle"
        manifest = build_release_bundle(corpus, benchmark, bundle)
        if from_hub:
            source = Path(
                hf_hub_download(
                    PILOT_REPO,
                    DATA_FILE,
                    repo_type="dataset",
                    revision=PILOT_REVISION,
                    cache_dir=scratch / "hub",
                    token=False,
                )
            )
        else:
            source = bundle / DATA_FILE
        jsonl_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if jsonl_digest != manifest["files"][DATA_FILE]:
            raise HubRoundtripError("JSONL bytes differ from the frozen pilot export")

        if from_hub:
            # Exercise the published card's config/split, not a replacement mapping.
            dataset = load_dataset(
                PILOT_REPO,
                name=CONFIG,
                split=SPLIT,
                revision=PILOT_REVISION,
                cache_dir=str(scratch / "datasets"),
                token=False,
            )
        else:
            with _offline_loading():
                dataset = load_dataset(
                    "json",
                    name=CONFIG,
                    data_files={SPLIT: str(source)},
                    split=SPLIT,
                    cache_dir=str(scratch / "datasets"),
                    token=False,
                )
        if dataset.info.config_name != CONFIG or str(dataset.split) != SPLIT:
            raise HubRoundtripError("unexpected loaded config or split")
        if dataset.num_rows != len(corpus["records"]):
            raise HubRoundtripError("loaded row count differs from the frozen pilot")
        if set(dataset.column_names) != set(corpus["records"][0]):
            raise HubRoundtripError("loaded columns differ from the frozen pilot")
        rebuilt = check_loaded_rows(corpus, list(dataset))

    results = evaluate_loaded_rows(rebuilt)
    return {
        "check": "pinned_pilot_roundtrip",
        "mode": "hub" if from_hub else "local_jsonl",
        "scope": "transport_fidelity_and_current_verifier_execution",
        "dataset": {
            "repo_id": PILOT_REPO if from_hub else None,
            "revision": PILOT_REVISION if from_hub else None,
            "config": CONFIG,
            "split": SPLIT,
            "data_file": DATA_FILE,
            "jsonl_sha256": jsonl_digest,
            "corpus_sha256": corpus_sha256(rebuilt),
            "record_count": len(results),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("pdecert", "datasets", "huggingface_hub", "pyarrow", "sympy")
            },
        },
        "configuration": dict(VERIFY_OPTIONS),
        "status_counts": dict(sorted(Counter(row["report"]["status"] for row in results).items())),
        "records": results,
        "limitations": [
            "No model inference, retraining, or new independent labels.",
            "Frozen symbolic-only pilot; not an Atlas v2 or callable loader.",
            "Hashes establish content identity, not correctness or authorship.",
            "Verifier reports retain their ordinary exact, empirical, or inconclusive scope.",
            "Not a reproduction of historical timings or a code/data release attestation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-hub", action="store_true", help="read the immutable public pilot anonymously"
    )
    parser.add_argument(
        "--output", type=Path, help="new JSON output path; existing files are refused"
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.output is not None and arguments.output.exists():
            raise HubRoundtripError("output already exists; choose a new path")
        result = run_smoke(from_hub=arguments.from_hub)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output is None:
            print(rendered, end="")
        else:
            with arguments.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
    except (ImportError, OSError, ValueError) as error:
        print(f"hub_pilot_smoke: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
