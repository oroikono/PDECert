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

1. Copy `review-template.json` to a private working file.
2. For each ID, read only its embedded `case` and raw generator transcript.
3. Check the candidate domain and regularity on the closed declared domain.
4. Differentiate the candidate and simplify each PDE residual directly.
5. substitute every initial and boundary trace.
6. Record a verdict, all demonstrated failure modes, and a short mathematical
   rationale containing the decisive identity or counterexample.
7. Do not inspect `results/provisional-review.json`, PDECert output, or baseline
   results until this pass is complete.

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

Review the resulting diff before replacing `corpus/pilot.json`. The importer
never overwrites the source corpus implicitly.

## Provisional file

`results/provisional-review.json` is machine-assisted preparation, not ground
truth and not a human annotation. It exists only to make post-review comparison
fast and auditable. Applying it without independently checking the mathematics
violates this protocol.
