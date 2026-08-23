# Roadmap

The plan is to keep each change small enough to understand, test, and review.
The days below are work sessions, not promises of artificial daily activity.

## First two weeks

- [x] Day 1: publish the three-state verifier, adversarial heat experiment, and tests.
- [ ] Day 2: define a JSON problem schema and reject malformed problems with clear errors.
- [ ] Day 3: add a command-line interface that verifies one problem and writes a JSON report.
- [ ] Day 4: add per-check time limits and record incomplete symbolic checks.
- [ ] Day 5: add explicit parameter assumptions and parameter-domain counterexamples.
- [ ] Day 6: support systems with multiple fields and coupled residual equations.
- [ ] Day 7: create a versioned candidate-corpus format with provenance fields.
- [ ] Day 8: collect the first 20 natural candidates from symbolic solvers and open models.
- [ ] Day 9: label those candidates manually and document the labeling protocol.
- [ ] Day 10: compare fixed collocation, SymPy, and PDECert on the natural candidates.

## Research milestones

- Publish a Hugging Face dataset containing problems, candidates, labels, and witnesses.
- Add interval-based counterexample search for supported expression classes.
- Distinguish classical, weak, and other solution semantics in the problem schema.
- Provide a small agent tool that returns either a certificate, a counterexample, or an
  inconclusive result.
- Connect counterexamples back to a candidate generator for verifier-guided repair.

The scope is intentionally conservative: supported classes should be explicit,
and unsupported cases should remain inconclusive.
