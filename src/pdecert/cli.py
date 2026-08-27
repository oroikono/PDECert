"""Command-line interface for PDECert."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .corpus import CorpusError, load_atlas_coverage, load_corpus_source
from .core import Status, verify
from .schema import SCHEMA_VERSION, SchemaError, load_case


EXIT_CODES = {
    Status.PROVED: 0,
    Status.REFUTED: 1,
    Status.INCONCLUSIVE: 2,
}
INPUT_ERROR = 64


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdecert",
        description="Check a symbolic PDE candidate without treating sampling as proof.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    verify_parser = subcommands.add_parser("verify", help="verify one JSON case")
    verify_parser.add_argument("case", type=Path, help="path to a versioned JSON case")
    verify_parser.add_argument("-o", "--output", type=Path, help="write the JSON report to a file")
    verify_parser.add_argument(
        "--tolerance",
        type=_positive_float,
        default=1e-9,
        help="counterexample residual threshold (default: 1e-9)",
    )
    verify_parser.add_argument(
        "--samples-per-axis",
        type=_positive_integer,
        default=5,
        help="off-grid samples per variable (default: 5)",
    )
    verify_parser.add_argument(
        "--symbolic-timeout",
        type=_positive_float,
        default=2.0,
        help="seconds allowed for each symbolic check (default: 2)",
    )
    verify_parser.add_argument(
        "--max-expression-ops",
        type=_positive_integer,
        default=10_000,
        help="maximum input operations admitted to a symbolic check (default: 10000)",
    )

    corpus_parser = subcommands.add_parser("corpus", help="inspect versioned PDE candidate corpora")
    corpus_commands = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    validate_parser = corpus_commands.add_parser(
        "validate", help="validate a corpus and summarize its coverage"
    )
    validate_parser.add_argument(
        "corpus",
        type=Path,
        help="path to a corpus JSON file or modular atlas directory",
    )
    return parser


def _run_verify(arguments: argparse.Namespace) -> int:
    try:
        case = load_case(arguments.case)
        report = verify(
            case.problem,
            case.candidate_fields,
            tolerance=arguments.tolerance,
            samples_per_axis=arguments.samples_per_axis,
            symbolic_timeout=arguments.symbolic_timeout,
            max_expression_ops=arguments.max_expression_ops,
        )
    except (OSError, SchemaError) as error:
        print(f"pdecert: {error}", file=sys.stderr)
        return INPUT_ERROR

    payload = {
        "problem": case.problem.name,
        "report": report.to_dict(),
        "schema_version": SCHEMA_VERSION,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        try:
            arguments.output.write_text(rendered)
        except OSError as error:
            print(f"pdecert: {error}", file=sys.stderr)
            return INPUT_ERROR
    return EXIT_CODES[report.status]


def _counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _run_corpus_validate(arguments: argparse.Namespace) -> int:
    try:
        corpus = load_corpus_source(arguments.corpus)
    except (OSError, CorpusError) as error:
        print(f"pdecert: {error}", file=sys.stderr)
        return INPUT_ERROR

    records = corpus["records"]
    verdicts = [
        record["annotation"]["verdict"]
        for record in records
        if record["annotation"]["verdict"] is not None
    ]
    failure_modes = [mode for record in records for mode in record["annotation"]["failure_modes"]]
    summary = {
        "annotation_statuses": _counts([record["annotation"]["status"] for record in records]),
        "corpus_version": corpus["corpus_version"],
        "failure_modes": _counts(failure_modes),
        "name": corpus["name"],
        "origin_kinds": _counts([record["origin"]["kind"] for record in records]),
        "records": len(records),
        "verdicts": _counts(verdicts),
    }
    coverage_path = arguments.corpus / "coverage.json"
    if arguments.corpus.is_dir() and coverage_path.exists():
        coverage = load_atlas_coverage(
            arguments.corpus,
            {record["id"] for record in records},
        )
        taxonomy = coverage["records"]
        summary.update(
            {
                "artifact_types": _counts(
                    [taxonomy[record["id"]]["artifact_type"] for record in records]
                ),
                "coverage_version": coverage["coverage_version"],
                "pde_families": _counts(
                    [
                        family
                        for record in records
                        for family in taxonomy[record["id"]]["pde_families"]
                    ]
                ),
                "spatial_dimensions": _counts(
                    [str(taxonomy[record["id"]]["spatial_dimension"]) for record in records]
                ),
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _run_verify(arguments)
    if arguments.command == "corpus" and arguments.corpus_command == "validate":
        return _run_corpus_validate(arguments)
    parser.error(f"unknown command: {arguments.command}")
    return INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
