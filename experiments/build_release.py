"""Build the digest-checked Hugging Face release bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pdecert import CorpusError, ReleaseError, build_release_bundle, load_corpus


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus/pilot.json"))
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("results/pilot-benchmark.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        benchmark = json.loads(arguments.benchmark.read_text())
        manifest = build_release_bundle(
            load_corpus(arguments.corpus),
            benchmark,
            arguments.output,
        )
    except json.JSONDecodeError as error:
        print(f"build_release: invalid benchmark JSON: {error.msg}", file=sys.stderr)
        return 2
    except (CorpusError, OSError, ReleaseError) as error:
        print(f"build_release: {error}", file=sys.stderr)
        return 2

    print(
        f"Wrote {manifest['record_count']}-record release bundle to {arguments.output} "
        f"({manifest['corpus_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
