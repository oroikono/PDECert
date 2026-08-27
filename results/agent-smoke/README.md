# Real-agent smoke results

These artifacts test the provider → agent → restricted tool → verifier path.
They are integration evidence, not model rankings or independent labels.

| Run | Model and provider | Verifier outcome | Observed interaction |
| --- | --- | --- | --- |
| `heat-qwen3-next-20260827-01` | `Qwen/Qwen3-Next-80B-A3B-Instruct` via Novita | `PROVED` with exact evidence | First tool payload rejected as malformed JSON; second payload materialized and satisfied the PDE, initial condition, and both boundary conditions. |

The [complete artifact](heat-qwen3-next-20260827-01.json) contains the exact
public prompt, raw response, tool payloads, tool feedback, Hub revision,
provider, requested decoding settings, package versions, and PDECert source
revision. The trusted case's reference field was held inside the verifier and
was not included in the model prompt.

Reproduce the integration path with an authenticated Hugging Face account:

```bash
python -m experiments.real_agent_smoke \
  --run-id <new-run-id> \
  --output results/agent-smoke/<new-run-id>.json
```

Use a new output path: the runner deliberately refuses to replace existing
evidence. Hosted inference is not bitwise reproducible; the recorded Hub commit
does not guarantee that a provider's serving stack or deployed weights remain
unchanged.
