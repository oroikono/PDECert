# Contributing

Thanks for taking an interest in PDECert. The project is still small, so focused
changes are easiest to review.

Before opening a pull request:

1. Explain the failure mode or verifier capability in an issue.
2. Keep the pull request to one main behavior.
3. Add a regression test or a minimal reproducible example.
4. Run `pytest` and `ruff check .` locally.
5. State what the result proves, and what it does not prove.

For soundness-sensitive changes, a new check must not turn finite numerical
sampling into a `PROVED` result. If the reasoning is incomplete, return
`INCONCLUSIVE` and record why.
