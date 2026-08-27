# LLM and scientific-agent integration

Agents generate or repair candidate artifacts. They are not verification
backends and their judgments are not independent labels. PDECert evaluates the
materialized artifact with the same verifier used for every other producer.

The framework-neutral layer has seven pieces:

- `AgentProposal` keeps raw model output, generator identity, metadata, repair
  parent, and the host-materialized artifact separate;
- `AgentEvaluation` pairs one proposal with machine verification evidence;
- `AgentTrace` records an ordered proposal → counterexample → repair history;
- `SymbolicAgentTool` accepts only candidate fields while holding the trusted
  PDE problem outside the agent-controlled payload.
- `SymbolicAgentSession` records accepted and rejected tool calls in order;
- `AgentRun` retains one model run without publishing raw text by default;
- `summarize_agent_runs` compares exact generator identities using verifier
  outcomes, call counts, and repair-to-`PROVED` counts.

Raw outputs are retained in memory and content-addressed with SHA-256. They are
excluded from serialized traces by default because prompts or model responses
may be large or sensitive. Passing `include_raw_outputs=True` includes them for
a deliberately public reproducibility artifact.

## Safe symbolic tool boundary

`SymbolicAgentTool` accepts a JSON object whose keys must exactly match the
trusted case fields. Values pass through PDECert's restricted expression parser.
The agent cannot replace the PDE, domain, parameters, or conditions, and the
tool never executes generated Python code. Invalid JSON, unknown fields,
unsupported syntax, and oversized payloads return structured errors.

```python
import json

from pdecert import SymbolicAgentTool

verifier = SymbolicAgentTool(trusted_case)
feedback = verifier(json.dumps({"u": "exp(-pi**2*t)*sin(pi*x)"}))
```

## Real smolagents runner

Install the optional integration and pass an initialized smolagents model:

```python
from smolagents import InferenceClientModel

from pdecert.integrations.smolagents import run_smolagents_symbolic_agent


model = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Instruct")
run = run_smolagents_symbolic_agent(
    trusted_case=trusted_case,
    model=model,
    prompt="Find and verify a symbolic solution.",
    run_id="heat-qwen-01",
    problem_id="heat-01",
    generator="Qwen/Qwen3-Next-80B-A3B-Instruct",
)
```

The runner constructs a real `ToolCallingAgent` and invokes `agent.run`, so API
calls are performed by the supplied model. The same entry point accepts local
or provider-backed smolagents models. Credentials remain under smolagents and
the provider SDK; PDECert does not read or serialize them.

The adapter deliberately uses `ToolCallingAgent`, not `CodeAgent`. The model can
only submit the declared JSON tool argument. `AgentRun` records every rejected
payload, counterexample, and repair, and hashes raw text in its public form.
Passing `include_raw_outputs=True` is required to serialize exact prompts or
responses.

Run the bounded provider-backed integration case with an existing Hugging Face
login:

```bash
python -m experiments.real_agent_smoke \
  --run-id heat-qwen3-next-20260827-01 \
  --output results/agent-smoke/heat-qwen3-next-20260827-01.json
```

The script checks that the requested inference provider is currently live,
records the full Hub revision, prompt, decoding request, package versions, raw
tool payloads, and final response, and refuses to overwrite an earlier result.
It never reads or serializes the Hugging Face token. A Hub revision does not
fully pin a hosted deployment, so every output carries that limitation. The
single heat-equation run is integration evidence, not a model benchmark.

```bash
pip install -e '.[agents]'
```

## Cross-model metrics

`summarize_agent_runs` groups runs by the exact `generator` string and reports:

- tool calls and rejected calls;
- materialized proposals;
- first-attempt and final `PROVED` rates;
- runs repaired from a non-`PROVED` first proposal to `PROVED`;
- final `PROVED`, `REFUTED`, `INCONCLUSIVE`, or `NO_VALID_PROPOSAL` counts.

These are verifier-grounded behavioral metrics. They are not called model
accuracy and do not replace independently labeled benchmark outcomes. Keep the
model revision, decoding parameters, prompt version, and seed in run metadata
when publishing comparisons.

## Materialized callable and PINN proposals

A host application may turn a model checkpoint or trusted model factory into a
`CallableCandidate`, preserve the unedited agent response in `raw_output`, and
call `evaluate_agent_proposal`. The ordinary autodiff verifier remains
empirical: sampled success is `INCONCLUSIVE`, while a sampled violation can be
`REFUTED` with a witness.

Symbolic proposals instead require a trusted `VerificationCase` whose original
constraint sources reference the declared field names. PDECert rematerializes
those constraints for every proposed expression. It rejects a case containing
only residuals previously substituted for another candidate, because reusing
such residuals would evaluate the wrong artifact.

## Deliberate exclusions

- PDECert does not choose a provider or model, and it never stores API keys.
- It does not treat an agent's self-critique as ground truth.
- It does not execute generated solver programs.
- It does not hide or rewrite the raw response during materialization.
- It does not claim that a repaired answer is correct until verification says
  what the available evidence establishes.

Generated solver programs remain a separate artifact boundary. They require a
documented process-isolation policy, resource limits, filesystem and network
policy, and output validation before execution can be enabled.

Run the framework-free example:

```bash
python -m examples.agent_repair_loop
```
