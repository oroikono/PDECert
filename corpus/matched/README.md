# Cross-artifact matched Atlas

This Atlas v2 preview binds two independently produced solution artifacts to
the same candidate-free Fisher--KPP problem template:

- the unedited Qwen3 symbolic response and its extracted field expression;
- a separately trained, frozen PINN with byte, configuration, source, and
  weight digests.

Both annotations remain `pending`. Atlas validation establishes content
identity, provenance structure, and template/artifact compatibility only. It
does not decide whether either artifact satisfies the PDE.

Validate the mixed corpus without installing PyTorch:

```bash
pdecert corpus validate corpus/matched
```

Reproduce the existing lane-specific evaluation with the optional autodiff
environment:

```bash
python -m experiments.trained_fisher_kpp_pair \
  --output results/trained-fisher-kpp-pair.json
```

The symbolic lane currently has exact evidence. The trained callable has a
replayable empirical counterexample at the declared tolerance. Those decisions
remain separate reports and are not human labels.
