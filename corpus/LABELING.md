# Human labeling protocol

The pilot corpus is not labeled until a person has reviewed all 20 cases. A
PDECert status, SymPy result, or provisional suggestion must not be copied into
`annotation` and described as independent human judgment.

## Verdicts

- `valid`: the candidate has the declared regularity and satisfies every PDE,
  initial condition, boundary condition, and parameter scope represented by the
  case;
- `invalid`: at least one represented obligation has a concrete violation;
- `unclear`: the written problem or required solution semantics are
  underspecified enough that validity cannot be decided.

For an invalid verdict, select every demonstrated failure mode. Do not infer a
failure mode merely because another checker reports one.

## Blind primary pass

The resumable card runner keeps the answer proposals hidden and saves every
decision to a git-ignored private directory:

```bash
python -m experiments.review_corpus
```

Alternatively, copy `review-template.json` to a private working file.

1. Start the card runner or private template.
2. For each ID, read only its embedded `case` and raw generator transcript.
3. Check the candidate domain and regularity on the closed declared domain.
4. Differentiate the candidate and simplify each PDE residual directly.
5. Substitute every initial and boundary trace.
6. Record a verdict, all demonstrated failure modes, and a short mathematical
   rationale containing the decisive identity or counterexample.
7. Do not inspect `results/provisional-review.json`, PDECert output, or baseline
   results until this pass is complete.

## Typed Atlas v2 review

Run the same blind workflow on symbolic/callable bundles with a private output:

```bash
python -m experiments.review_corpus corpus/matched \
  --output private-reviews/matched-review.json
```

The version 2 review binds itself to a canonical SHA-256 digest of the loaded
Atlas and adds an explicit basis to every completed decision. The runner omits
coverage labels, training losses, PDECert reports, and baseline outcomes.

Allowed bases are deliberately conservative:

- `manual_derivation` may support a symbolic decision when the reviewer checks
  the represented identities directly;
- `independent_counterexample` may support an invalid decision when the
  rationale gives a replayable violation obtained independently of PDECert;
- `rigorous_external_certificate` may support a decision only when its artifact,
  obligations, domain, norm, assumptions, and constants match this record; and
- `scope_assessment` supports `unclear` when the available semantics or evidence
  cannot justify a binary decision.

A callable `valid` verdict requires a matching rigorous external certificate.
Architecture inspection, low training loss, agreement with a finite reference
grid, or a passing autodiff sample cannot prove that a callable satisfies every
classical strong-form obligation. If no independent counterexample or rigorous
certificate is available, use `unclear`; do not copy the existing PDECert result
into the human annotation.

## Comparison and disagreement

After the blind pass, compare the review with the provisional file and PDECert
report. A disagreement is not automatically reviewer error. Recompute the
specific obligation, note the competing conclusions, and ask a second reviewer
to decide without seeing the first reviewer's identity. An adjudicated record
must contain at least two annotator identifiers and the final rationale.

## Import

The importer refuses incomplete IDs, null verdicts, empty rationales, invalid
labels without a failure mode, and any run that does not explicitly confirm an
independent human review.

```bash
python -m experiments.apply_review my-completed-review.json \
  --annotator YOUR_PUBLIC_ID \
  --confirm-independent-review \
  --output corpus/pilot-labeled.json
```

For a modular Atlas, provide the directory as `--corpus` and a new directory as
`--output`:

```bash
python -m experiments.apply_review private-reviews/community-review.json \
  --corpus corpus/community \
  --annotator YOUR_PUBLIC_ID \
  --confirm-independent-review \
  --output private-reviews/community-labeled
```

The modular importer writes a complete new Atlas atomically and refuses an
existing destination. It does not alter the source Atlas or the stored raw
generator outputs.

For Atlas v2, the importer also verifies the source digest, retains the declared
review basis in each annotation, and copies every template, artifact, raw output,
integrity record, coverage file, and README byte unchanged. Only `record.json`
annotations are rewritten in the new directory.

Review the resulting file or directory before making it canonical. The importer
never overwrites the source corpus or Atlas implicitly.

## Provisional file

`results/provisional-review.json` is machine-assisted preparation, not ground
truth and not a human annotation. It exists only to make post-review comparison
fast and auditable. Applying it without independently checking the mathematics
violates this protocol.
