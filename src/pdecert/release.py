"""Deterministic release bundles for the labeled PDECert pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .benchmark import BENCHMARK_VERSION, OUTCOMES, corpus_sha256
from .corpus import validate_corpus


RELEASE_VERSION = 1
METHODS = ("fixed_collocation", "pdecert", "sympy_residual")
METHOD_TITLES = {
    "fixed_collocation": "Fixed collocation",
    "pdecert": "PDECert",
    "sympy_residual": "Direct SymPy residual",
}


class ReleaseError(ValueError):
    """Raised when labeled data and a report cannot form a valid release."""


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseError(f"{path}: expected an object")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _expected_metrics(records: list[Mapping[str, object]]) -> dict[str, object]:
    valid = [record for record in records if record["truth"] == "valid"]
    invalid = [record for record in records if record["truth"] == "invalid"]
    decisive = [record for record in records if record["outcome"] != "inconclusive"]
    correct = [
        record
        for record in records
        if (record["truth"], record["outcome"]) in {("valid", "accept"), ("invalid", "reject")}
    ]
    false_acceptances = [record for record in invalid if record["outcome"] == "accept"]
    false_rejections = [record for record in valid if record["outcome"] == "reject"]
    inconclusive = [record for record in records if record["outcome"] == "inconclusive"]
    witnessed_invalid = [
        record
        for record in invalid
        if record["outcome"] == "reject" and record.get("witness") is not None
    ]
    return {
        "accuracy": _rate(len(correct), len(records)),
        "correct_count": len(correct),
        "decisive_accuracy": _rate(len(correct), len(decisive)),
        "decisive_count": len(decisive),
        "false_acceptance_count": len(false_acceptances),
        "false_acceptance_rate": _rate(len(false_acceptances), len(invalid)),
        "false_rejection_count": len(false_rejections),
        "false_rejection_rate": _rate(len(false_rejections), len(valid)),
        "inconclusive_count": len(inconclusive),
        "inconclusive_rate": _rate(len(inconclusive), len(records)),
        "invalid_count": len(invalid),
        "invalid_witness_count": len(witnessed_invalid),
        "invalid_witness_rate": _rate(len(witnessed_invalid), len(invalid)),
        "scored_count": len(records),
        "valid_count": len(valid),
    }


def validate_release_inputs(corpus: object, benchmark: object) -> None:
    """Require completed labels and a benchmark bound to exactly that corpus."""

    validate_corpus(corpus)
    corpus_object = _mapping(corpus, "corpus")
    corpus_records = corpus_object["records"]
    pending = [
        record["id"] for record in corpus_records if record["annotation"]["status"] == "pending"
    ]
    if pending:
        raise ReleaseError(
            "release requires completed human labels; pending: " + ", ".join(pending)
        )
    scored_records = [
        record
        for record in corpus_records
        if record["annotation"]["verdict"] in {"valid", "invalid"}
    ]
    if not scored_records:
        raise ReleaseError("release requires at least one valid or invalid record")

    report = _mapping(benchmark, "benchmark")
    required = {
        "benchmark_version",
        "configuration",
        "corpus",
        "environment",
        "method_definitions",
        "methods",
    }
    if set(report) != required:
        raise ReleaseError("benchmark: fields do not match benchmark version 1")
    if (
        isinstance(report["benchmark_version"], bool)
        or report["benchmark_version"] != BENCHMARK_VERSION
    ):
        raise ReleaseError(f"benchmark_version must be {BENCHMARK_VERSION}")
    try:
        json.dumps(report, allow_nan=False)
        json.dumps(corpus_object, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReleaseError(f"release input is not strict JSON: {error}") from error

    configuration = _mapping(report["configuration"], "benchmark.configuration")
    expected_configuration = {
        "method_order",
        "points_per_axis",
        "symbolic_timeout_seconds",
        "timing_note",
        "tolerance",
    }
    if set(configuration) != expected_configuration:
        raise ReleaseError("benchmark configuration fields are unsupported")
    if configuration["method_order"] != list(METHODS):
        raise ReleaseError("benchmark method_order does not match the release specification")
    points_per_axis = configuration["points_per_axis"]
    if (
        isinstance(points_per_axis, bool)
        or not isinstance(points_per_axis, int)
        or points_per_axis < 2
    ):
        raise ReleaseError("benchmark points_per_axis is invalid")
    tolerance = configuration["tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ReleaseError("benchmark tolerance is invalid")
    symbolic_timeout = configuration["symbolic_timeout_seconds"]
    if symbolic_timeout is not None and (
        isinstance(symbolic_timeout, bool)
        or not isinstance(symbolic_timeout, (int, float))
        or not math.isfinite(symbolic_timeout)
        or symbolic_timeout <= 0
    ):
        raise ReleaseError("benchmark symbolic timeout is invalid")
    if (
        not isinstance(configuration["timing_note"], str)
        or not configuration["timing_note"].strip()
    ):
        raise ReleaseError("benchmark timing note is invalid")
    environment = _mapping(report["environment"], "benchmark.environment")
    if set(environment) != {"pdecert", "platform", "python", "sympy"} or any(
        not isinstance(value, str) or not value.strip() for value in environment.values()
    ):
        raise ReleaseError("benchmark environment fields are invalid")
    definitions = _mapping(report["method_definitions"], "benchmark.method_definitions")
    if set(definitions) != set(METHODS) or any(
        not isinstance(value, str) or not value.strip() for value in definitions.values()
    ):
        raise ReleaseError("benchmark method definitions are invalid")

    report_corpus = _mapping(report["corpus"], "benchmark.corpus")
    if set(report_corpus) != {
        "excluded_unclear",
        "name",
        "scored_records",
        "sha256",
        "total_records",
    }:
        raise ReleaseError("benchmark corpus summary fields are unsupported")
    expected_digest = corpus_sha256(corpus)
    if report_corpus.get("sha256") != expected_digest:
        raise ReleaseError("benchmark corpus digest does not match the release corpus")
    if report_corpus.get("name") != corpus_object["name"]:
        raise ReleaseError("benchmark corpus name does not match the release corpus")
    if isinstance(report_corpus.get("total_records"), bool) or report_corpus.get(
        "total_records"
    ) != len(corpus_records):
        raise ReleaseError("benchmark total_records does not match the release corpus")
    if isinstance(report_corpus.get("scored_records"), bool) or report_corpus.get(
        "scored_records"
    ) != len(scored_records):
        raise ReleaseError("benchmark scored_records does not match the release corpus")
    if isinstance(report_corpus.get("excluded_unclear"), bool) or report_corpus.get(
        "excluded_unclear"
    ) != len(corpus_records) - len(scored_records):
        raise ReleaseError("benchmark excluded_unclear does not match the release corpus")

    methods = _mapping(report["methods"], "benchmark.methods")
    if set(methods) != set(METHODS):
        raise ReleaseError("benchmark methods do not match the release specification")
    expected_ids = [record["id"] for record in scored_records]
    expected_truth = {record["id"]: record["annotation"]["verdict"] for record in scored_records}
    for method_name in METHODS:
        method = _mapping(methods[method_name], f"benchmark.methods.{method_name}")
        if set(method) != {"metrics", "records", "runtime_seconds"}:
            raise ReleaseError(f"benchmark method has unsupported fields: {method_name}")
        method_records = method["records"]
        if not isinstance(method_records, list) or any(
            not isinstance(record, Mapping) for record in method_records
        ):
            raise ReleaseError(f"benchmark method records are invalid: {method_name}")
        actual_ids = [record.get("id") for record in method_records]
        if actual_ids != expected_ids:
            raise ReleaseError(f"benchmark method IDs do not match corpus order: {method_name}")
        for record in method_records:
            record_id = record["id"]
            runtime = record.get("runtime_seconds")
            if (
                isinstance(runtime, bool)
                or not isinstance(runtime, (int, float))
                or not math.isfinite(runtime)
                or runtime < 0
            ):
                raise ReleaseError(f"benchmark runtime is invalid: {method_name}/{record_id}")
            if record.get("truth") != expected_truth[record_id]:
                raise ReleaseError(
                    f"benchmark truth does not match corpus: {method_name}/{record_id}"
                )
            if record.get("outcome") not in OUTCOMES:
                raise ReleaseError(f"benchmark outcome is invalid: {method_name}/{record_id}")
        expected_metrics = _expected_metrics(method_records)
        if json.dumps(method["metrics"], sort_keys=True) != json.dumps(
            expected_metrics, sort_keys=True
        ):
            raise ReleaseError(f"benchmark metrics do not match record outcomes: {method_name}")
        total_runtime = method["runtime_seconds"]
        if (
            isinstance(total_runtime, bool)
            or not isinstance(total_runtime, (int, float))
            or not math.isfinite(total_runtime)
            or total_runtime < 0
        ):
            raise ReleaseError(f"benchmark total runtime is invalid: {method_name}")


def _percentage(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _runtime(value: object) -> str:
    return f"{float(value):.4f} s"


def _render_card(corpus: Mapping[str, object], benchmark: Mapping[str, object]) -> str:
    records = corpus["records"]
    verdicts = Counter(record["annotation"]["verdict"] for record in records)
    origins = Counter(record["origin"]["kind"] for record in records)
    licenses = sorted(
        {
            (record["origin"]["producer"], record["origin"]["license"] or "not specified")
            for record in records
        }
    )
    metrics_rows = []
    for method_name in METHODS:
        result = benchmark["methods"][method_name]
        metrics = result["metrics"]
        metrics_rows.append(
            "| {method} | {accuracy} | {false_accept} | {false_reject} | "
            "{inconclusive} | {witness} | {runtime} |".format(
                method=METHOD_TITLES[method_name],
                accuracy=_percentage(metrics["accuracy"]),
                false_accept=_percentage(metrics["false_acceptance_rate"]),
                false_reject=_percentage(metrics["false_rejection_rate"]),
                inconclusive=_percentage(metrics["inconclusive_rate"]),
                witness=_percentage(metrics["invalid_witness_rate"]),
                runtime=_runtime(result["runtime_seconds"]),
            )
        )
    license_rows = "\n".join(
        f"- {producer}: `{license_name}`" for producer, license_name in licenses
    )
    environment = benchmark["environment"]
    corpus_digest = benchmark["corpus"]["sha256"]
    return f"""---
license: other
license_name: mixed-permissive
license_link: https://github.com/oroikono/PDECert/blob/main/corpus/README.md
language:
- en
pretty_name: PDECert Natural-Candidate Pilot
size_categories:
- n<1K
tags:
- partial-differential-equations
- symbolic-computation
- verification
configs:
- config_name: default
  data_files:
  - split: test
    path: data/pilot.jsonl
---

# PDECert Natural-Candidate Pilot

This is a provenance-bearing pilot benchmark for checking symbolic candidate
solutions to partial differential equations. Each row contains the unedited
generator output, a fully instantiated verification case, content digest,
producer metadata, and completed human annotation.

## Dataset summary

- Records: {len(records)}
- Symbolic-solver outputs: {origins["symbolic_solver"]}
- Open-model outputs: {origins["open_model"]}
- Valid: {verdicts["valid"]}
- Invalid: {verdicts["invalid"]}
- Unclear: {verdicts["unclear"]}
- Corpus SHA-256: `{corpus_digest}`

The `test` split is the complete pilot. It is an evaluation set, not training
data. Load it from the Hub with:

```python
from datasets import load_dataset

pilot = load_dataset("oroikono/pdecert-pilot", split="test")
```

## Relationship to the PDECert library

This dataset is the frozen symbolic-only pilot released with PDECert 0.1.0.
The current library also has separate differentiable callable/PINN and agent-
trace paths, but those artifact types are not represented by these 20 rows and
their evidence must not be inferred from this benchmark. Install the latest
prerelease with `python -m pip install --pre pdecert` and read the
[evidence contract](https://github.com/oroikono/PDECert/blob/main/LIMITATIONS_AND_THREATS_TO_VALIDITY.md)
before interpreting a result.

## Benchmark

| Method | Accuracy | False accept | False reject | Inconclusive | Invalid witness | Runtime |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics_rows)}

False-acceptance rates use invalid records as the denominator; false-rejection
rates use valid records. Accuracy counts inconclusive outcomes as incorrect.
Witness rate is the fraction of invalid records rejected with a concrete PDECert
witness. Full per-record outcomes and descriptive single-process timings are in
`results/pilot-benchmark.json`.

Environment: PDECert {environment["pdecert"]}, Python {environment["python"]},
SymPy {environment["sympy"]}, `{environment["platform"]}`.

## Collection and annotation

The pilot contains real outputs collected from ten SymPy `pdsolve` calls and
ten generations from the pinned open model recorded in each row. Raw output was
not reconstructed or polished. Human verdicts follow the blind primary-review
and disagreement procedure documented in the
[source repository](https://github.com/oroikono/PDECert/blob/main/corpus/LABELING.md).

## Row structure

- `id`: stable lowercase record identifier;
- `raw_output` and `output_sha256`: unedited output and its content digest;
- `origin`: producer, version or revision, source, license, generation time, and input;
- `case`: candidate fields, domains, residuals, conditions, and schema version;
- `annotation`: verdict, failure modes, rationale, status, and reviewer identifiers.

## Licensing

The PDECert packaging and original annotations are released with the source
project under MIT. The source software and models carry the licenses recorded
in each row:

{license_rows}

An origin license does not by itself settle every right in generated output.
Consumers should inspect each row's `origin.license`, `origin.source_url`, and
the applicable provider terms. The dataset card uses `license: other` because
the bundle has mixed provenance rather than pretending every component shares
one license.

## Limitations and responsible use

This is a deliberately small, constructed pilot with a designed 10/10 producer
balance. It is not a random or representative sample of PDE solver or language
model behavior, and aggregate percentages must not be generalized to other
models, equations, prompts, or solution semantics. The cases encode classical
pointwise residual and trace obligations; they do not establish weak, viscosity,
distributional, or numerical-solution validity. A machine result is not a
substitute for mathematical review in safety-critical settings.

## Reproducibility

The release manifest hashes the dataset card, JSONL data, and benchmark report.
The benchmark report is accepted into the bundle only when its embedded corpus
digest, record IDs, truth labels, and recomputed metrics agree with the exact
JSONL rows. Source, tests, and collection scripts are available at
[oroikono/PDECert](https://github.com/oroikono/PDECert).
"""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_release_bundle(
    corpus: object,
    benchmark: object,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write a deterministic Hugging Face-ready bundle and return its manifest."""

    validate_release_inputs(corpus, benchmark)
    corpus_object = _mapping(corpus, "corpus")
    benchmark_object = _mapping(benchmark, "benchmark")
    output = Path(output_directory)
    if output.exists() and not output.is_dir():
        raise ReleaseError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ReleaseError(f"output directory is not empty: {output}")

    jsonl = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in corpus_object["records"]
    ).encode()
    report = (
        json.dumps(benchmark_object, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    card = _render_card(corpus_object, benchmark_object).encode()
    contents = {
        "README.md": card,
        "data/pilot.jsonl": jsonl,
        "results/pilot-benchmark.json": report,
    }
    verdict_counts = Counter(record["annotation"]["verdict"] for record in corpus_object["records"])
    manifest: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "corpus_sha256": corpus_sha256(corpus),
        "files": {name: _sha256(content) for name, content in sorted(contents.items())},
        "record_count": len(corpus_object["records"]),
        "release_version": RELEASE_VERSION,
        "verdict_counts": dict(sorted(verdict_counts.items())),
    }
    contents["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()

    output.mkdir(parents=True, exist_ok=True)
    for relative_path, content in contents.items():
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return manifest
