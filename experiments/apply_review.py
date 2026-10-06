"""Apply a completed independent review to a candidate corpus."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pdecert import (
    CROSS_ARTIFACT_ATLAS_VERSION,
    ReviewError,
    apply_review,
    dump_atlas,
    dump_cross_artifact_atlas,
    dump_corpus,
    load_atlas_coverage,
    load_corpus_source,
)


def _dump_new_corpus(value: object, destination: Path) -> None:
    """Publish a complete corpus without replacing an existing path."""

    if destination.exists() or destination.is_symlink():
        raise ReviewError(f"{destination}: refusing to overwrite an existing path")
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".pdecert-review-"
    ) as directory:
        staged = Path(directory) / "corpus.json"
        dump_corpus(value, staged)
        try:
            # Unlike replace(), link() refuses even a concurrently created destination.
            os.link(staged, destination)
        except FileExistsError as exc:
            raise ReviewError(f"{destination}: refusing to overwrite an existing path") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--confirm-independent-review", action="store_true")
    arguments = parser.parse_args(argv)

    review = json.loads(arguments.review.read_text())
    corpus = load_corpus_source(arguments.corpus)
    labeled = apply_review(
        corpus,
        review,
        annotator=arguments.annotator,
        confirmed_independent_review=arguments.confirm_independent_review,
    )
    if corpus.get("atlas_version") == CROSS_ARTIFACT_ATLAS_VERSION:
        dump_cross_artifact_atlas(labeled, arguments.corpus, arguments.output)
        kind = "typed atlas"
    elif arguments.corpus.is_dir():
        coverage_path = arguments.corpus / "coverage.json"
        coverage = (
            load_atlas_coverage(
                arguments.corpus,
                {record["id"] for record in corpus["records"]},
            )
            if coverage_path.exists()
            else None
        )
        dump_atlas(labeled, arguments.output, coverage=coverage)
        kind = "atlas"
    else:
        _dump_new_corpus(labeled, arguments.output)
        kind = "corpus"
    print(f"Wrote labeled {kind} to {arguments.output}")


if __name__ == "__main__":
    main()
