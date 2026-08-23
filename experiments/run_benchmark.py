"""Run the label-gated benchmark and write its JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pdecert import BenchmarkError, CorpusError, evaluate_corpus, load_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-axis", type=int, default=5)
    parser.add_argument("--symbolic-timeout", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    arguments = parser.parse_args(argv)

    try:
        report = evaluate_corpus(
            load_corpus(arguments.corpus),
            points_per_axis=arguments.points_per_axis,
            symbolic_timeout=arguments.symbolic_timeout,
            tolerance=arguments.tolerance,
        )
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    except (BenchmarkError, CorpusError, OSError) as error:
        print(f"run_benchmark: {error}", file=sys.stderr)
        return 2
    print(f"Wrote benchmark report to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
