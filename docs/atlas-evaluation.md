# Evaluating typed Atlas records

Atlas v2 stores symbolic expressions and restricted frozen callables beside one
candidate-free problem template. The evaluation interface runs those stored
artifacts through their compatible PDECert backends while keeping every result
separate:

```bash
pip install -e ".[autodiff]"
pdecert corpus evaluate corpus/matched \
  --callable-tolerance 0.001 \
  --samples-per-axis 6 \
  --output matched-evaluation.json
```

The command validates the complete Atlas before materializing a callable. It
then binds symbolic fields through the restricted parser or materializes the
bounded dense-tanh callable format and compiles the trusted template for
automatic differentiation. It does not execute generated source code.

Use `--record` repeatedly to evaluate a subset. A symbolic-only selection works
with the core installation and does not import PyTorch:

```bash
pdecert corpus evaluate corpus/matched \
  --record qwen3-fisher-kpp-01
```

## Evidence contract

The version-1 evaluation document has `evidence_policy` set to
`per_record_no_aggregation`. It intentionally has no top-level status or
decision-evidence field.

- A symbolic record may be `PROVED` only within the existing restricted exact
  backend and its represented classical strong-form obligations. That decision
  applies to the parsed fields in `artifact.json`; it does not prove that their
  extraction preserved the meaning of the raw model response.
- A callable record may be `REFUTED` by a replayable sampled witness.
- A callable whose finite samples pass remains `INCONCLUSIVE`.
- Evidence never transfers between records, even when they share `problem_id`.
- Atlas annotations are not copied into the evaluation rows and are not used to
  choose a machine result.

The public schema also binds each artifact type to its evaluator. Callable
rows reject proof statuses, exact or rigorous decision evidence, and
non-empirical evidence events. An Atlas with no selected records is rejected
rather than producing an empty document that looks like a completed run.

Exit code `0` means that the requested records were evaluated and the complete
document was produced. It does not mean that every record was proved. Invalid
input, unsupported Atlas versions, missing record IDs, or a missing optional
autodiff dependency return exit code `64`.

## Reproduction and identity

Every output records:

- evaluation format version and the `per_record_no_aggregation` policy;
- the canonical SHA-256 of the loaded Atlas;
- selected record, problem, artifact, and evaluator identities;
- symbolic and callable tolerances, sampling density, symbolic timeout, and
  structural expression budget;
- Python, PDECert, SymPy, and, when used, PyTorch versions; and
- the complete obligation-level PDECert report and witness.

The Atlas digest is the same digest used by typed blind review. It covers the
loaded manifest and records, including candidate, problem, provenance,
annotation, and digest-bound artifact contents. Review-neutral `README.md` and
`coverage.json` bytes are excluded. A matching digest establishes content
identity only; it does not establish authorship, trusted execution, training
provenance, correctness, or an independent label.

The public JSON shape is
[`schema/atlas-evaluation-v1.schema.json`](../schema/atlas-evaluation-v1.schema.json).

## Current boundary

The runner accepts Atlas v2 `symbolic_expression` and `callable_model` records
whose templates use `classical_strong` semantics. Atlas v1, numerical fields,
generated programs, arbitrary checkpoints, weak or entropy solutions, global
neural residual bounds, and solution-error guarantees remain unsupported. This
is a reusable evaluator, not yet a baseline comparison, labeled benchmark, or
immutable mixed-corpus release pipeline.
