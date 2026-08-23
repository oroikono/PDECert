"""Apply a completed independent review to a candidate corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdecert import apply_review, dump_corpus, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--confirm-independent-review", action="store_true")
    arguments = parser.parse_args()

    review = json.loads(arguments.review.read_text())
    labeled = apply_review(
        load_corpus(arguments.corpus),
        review,
        annotator=arguments.annotator,
        confirmed_independent_review=arguments.confirm_independent_review,
    )
    dump_corpus(labeled, arguments.output)
    print(f"Wrote labeled corpus to {arguments.output}")


if __name__ == "__main__":
    main()
