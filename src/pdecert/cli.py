"""Command-line interface for PDECert."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

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
    return parser


def _run_verify(arguments: argparse.Namespace) -> int:
    try:
        case = load_case(arguments.case)
        report = verify(
            case.problem,
            case.candidate_expressions,
            tolerance=arguments.tolerance,
            samples_per_axis=arguments.samples_per_axis,
            symbolic_timeout=arguments.symbolic_timeout,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        return _run_verify(arguments)
    parser.error(f"unknown command: {arguments.command}")
    return INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
