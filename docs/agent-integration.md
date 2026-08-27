# LLM and scientific-agent integration

Agents generate or repair candidate artifacts. They are not verification
backends and their judgments are not independent labels. PDECert evaluates the
materialized artifact with the same verifier used for every other producer.

The initial framework-neutral scaffold has four pieces:

- `AgentProposal` keeps raw model output, generator identity, metadata, repair
  parent, and the host-materialized artifact separate;
- `AgentEvaluation` pairs one proposal with machine verification evidence;
- `AgentTrace` records an ordered proposal → counterexample → repair history;
- `SymbolicAgentTool` accepts only candidate fields while holding the trusted
  PDE problem outside the agent-controlled payload.

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

The callable can be wrapped without adding an agent framework to PDECert's core
dependencies. For example, an application using smolagents can expose it through
that framework's ordinary tool decorator:

```python
from smolagents import tool


@tool
def verify_pde_candidate(candidate_fields_json: str) -> str:
    """Verify candidate fields against the trusted PDE specification."""
    return verifier(candidate_fields_json)
```

Equivalent wrappers can be written for other agent frameworks.

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

- PDECert does not call an LLM API or choose a model.
- It does not treat an agent's self-critique as ground truth.
- It does not execute generated solver programs.
- It does not hide or rewrite the raw response during materialization.
- It does not claim that a repaired answer is correct until verification says
  what the available evidence establishes.

`ProgramCandidate` requires a documented process-isolation boundary, resource
limits, filesystem and network policy, and output validation before it can be
added safely.

Run the framework-free example:

```bash
python -m examples.agent_repair_loop
```
