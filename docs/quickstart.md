# Five-minute offline quickstart

This demonstration exercises PDECert's three-way decision and a recorded
proposal-repair trace without cloning the repository, downloading benchmark
assets, contacting a model provider, or installing an agent framework.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pdecert==0.1.1rc2
pdecert quickstart
pdecert quickstart --json > pdecert-quickstart.json
```

The command checks one trusted classical heat-equation problem against three
symbolic artifacts:

1. An exact solution receives `PROVED` with `EXACT` decision evidence.
2. A candidate that satisfies the PDE but violates its initial and right
   boundary data receives `REFUTED` with an empirical, replayable witness.
3. A real error below the configured floating-point tolerance produces sampled
   passes but remains `INCONCLUSIVE`.

The same rejected and repaired candidates form a framework-neutral agent trace:
`attempt-1` is `REFUTED`, and its linked `attempt-2` repair is `PROVED`. These
are deterministic recorded fixtures, not claims about a live language model.
PDECert hashes the retained raw proposal text and excludes it from the default
JSON trace; the verifier report remains separate from proposal provenance.

The JSON output contains the complete versioned reports, evidence events,
counterexample, proposal digests, parent link, evaluation settings, and
quickstart self-checks. A successful demonstration exits with code `0`. Exit
code `70` means an expected outcome changed, which makes the command suitable
as a release or installation smoke test.

This command demonstrates the evidence contract, not broad PDE coverage or
model quality. The exact result applies only to the represented classical
strong-form obligations. The sampled pass is explicitly not a certificate. See
the project-wide
[limitations and threats-to-validity statement](../LIMITATIONS_AND_THREATS_TO_VALIDITY.md)
before using a report in an evaluation or publication.
