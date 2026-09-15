"""Recount corpus annotation coverage without evaluating candidates or inferring accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pdecert import CorpusError, corpus_sha256, cross_artifact_atlas_sha256, load_corpus_source
from pdecert.corpus import ANNOTATION_STATUSES, VERDICTS


def audit_benchmark_coverage(path: str | Path) -> dict[str, Any]:
    """Load a supported corpus or Atlas and describe its stored annotations.

    All counts use candidate records, not independent mathematical problems.
    Annotator identifiers and declared review status do not establish independent
    review. The existing loaders validate records before they are counted.
    """

    corpus = load_corpus_source(path)
    records = corpus["records"]
    by_origin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_annotator: dict[str, list[str]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    annotator_counts = {"zero": 0, "one": 0, "two_or_more": 0}
    binary_by_origin: dict[str, set[str]] = defaultdict(set)
    binary_count = 0
    for record in records:
        annotation = record["annotation"]
        origin = record["origin"]["kind"]
        by_origin[origin].append(record)
        statuses[annotation["status"]] += 1
        annotators = annotation["annotators"]
        count_bucket = (
            "zero" if not annotators else "one" if len(annotators) == 1 else "two_or_more"
        )
        annotator_counts[count_bucket] += 1
        for annotator in annotators:
            by_annotator[annotator].append(record["id"])
        if annotation["verdict"] in {"valid", "invalid"}:
            binary_by_origin[origin].add(annotation["verdict"])
            binary_count += 1

    origin_rows = []
    for origin, origin_records in sorted(by_origin.items()):
        verdicts = Counter(
            record["annotation"]["verdict"] or "pending" for record in origin_records
        )
        producers = Counter(
            (record["origin"]["producer"], record["origin"]["identifier"])
            for record in origin_records
        )
        origin_rows.append(
            {
                "origin_kind": origin,
                "record_count": len(origin_records),
                "record_ids": sorted(record["id"] for record in origin_records),
                "verdict_counts": {
                    verdict: verdicts[verdict] for verdict in sorted(VERDICTS | {"pending"})
                },
                "producer_identifiers": [
                    {"producer": producer, "identifier": identifier, "record_count": count}
                    for (producer, identifier), count in sorted(producers.items())
                ],
            }
        )

    binary_verdicts = {verdict for values in binary_by_origin.values() for verdict in values}
    # With only one observed binary verdict or origin there is no two-sided comparison.
    comparable = len(binary_verdicts) == 2 and len(binary_by_origin) >= 2
    separation = (
        all(len(values) == 1 for values in binary_by_origin.values()) if comparable else None
    )
    is_typed = corpus.get("atlas_version") == 2
    digest = cross_artifact_atlas_sha256(corpus) if is_typed else corpus_sha256(corpus)
    return {
        "coverage_audit_version": 1,
        "scope": "descriptive_stored_annotations_only",
        "unit": "candidate_record",
        "corpus": {
            "name": corpus["name"],
            "sha256": digest,
            "digest_kind": "canonical_loaded_atlas_v2" if is_typed else "canonical_loaded_corpus",
            "record_count": len(records),
        },
        "origin_grouping": "origin.kind",
        "origin_verdict_counts": origin_rows,
        "annotation_coverage": {
            "status_counts": {status: statuses[status] for status in sorted(ANNOTATION_STATUSES)},
            "completed_record_count": len(records) - statuses["pending"],
            "pending_record_count": statuses["pending"],
        },
        "reviewer_coverage": {
            "unique_annotator_id_count": len(by_annotator),
            "record_counts_by_annotator_id_count": annotator_counts,
            "record_ids_by_annotator": {
                annotator: sorted(ids) for annotator, ids in sorted(by_annotator.items())
            },
            "independent_review_status": "not_established_by_annotation_metadata",
            "independent_reviewer_count": None,
        },
        "origin_verdict_diagnostic": {
            "binary_record_count": binary_count,
            "excluded_pending_or_unclear_record_count": len(records) - binary_count,
            "origins_with_both_binary_verdicts": sorted(
                origin for origin, values in binary_by_origin.items() if len(values) == 2
            ),
            "single_binary_verdict_by_origin": {
                origin: next(iter(values))
                for origin, values in sorted(binary_by_origin.items())
                if len(values) == 1
            },
            "origin_determines_binary_verdict": separation,
        },
        "limitations": [
            "Counts describe stored annotations, not accuracy or independent ground truth.",
            "Annotator IDs, even multiple IDs or adjudicated status, do not establish "
            "independent people, blindness, affiliation, or independent mathematical review.",
            "Origin kinds are coarse groups; producer and identifier are declared metadata, "
            "not evidence of independent generators or representative model diversity.",
            "Origin separation describes completed valid/invalid labels only. A false value "
            "does not rule out confounding; null means too few origins or binary verdicts.",
            "Pending and unclear records remain in coverage counts and are not binary labels.",
            "Records may share problems or artifacts; counts are not independent sample sizes. "
            "Audit each source separately because corpora can overlap.",
            "Loading validates content and representation; no candidate evaluation, training, "
            "model inference, or review is performed. Digests establish content identity only.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("corpus/pilot.json"))
    arguments = parser.parse_args(argv)
    try:
        report = audit_benchmark_coverage(arguments.corpus)
    except (CorpusError, OSError, UnicodeError) as error:
        print(f"audit_benchmark_coverage: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
