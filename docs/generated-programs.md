# Generated solver programs

`ProgramCandidate` stores untrusted Python source without running it. The source
must emit exactly one JSON object whose keys are the declared fields and whose
values are symbolic expression strings.

```python
from pdecert import ProgramCandidate, execute_program_candidate

candidate = ProgramCandidate(
    source='import json; print(json.dumps({"u": "sin(pi*x)"}))',
    declared_field_names=("u",),
)

# Fails before candidate.source is executed because no sandbox is configured.
execute_program_candidate(candidate)
```

Execution requires a separate object implementing `ProgramSandbox`. Its
capabilities must include process, network, filesystem, resource, and secret
isolation. The backend receives `ProgramLimits` and returns a bounded
`SandboxResult`. PDECert has no local subprocess or `exec` fallback.

After a successful isolated run, materialize the output through the same
restricted symbolic grammar used by agent proposals:

```python
from pdecert import AgentProposal, SymbolicAgentTool, evaluate_agent_proposal

output = execute_program_candidate(candidate, audited_sandbox)
artifact = output.materialize(SymbolicAgentTool(trusted_case))
evaluation = evaluate_agent_proposal(
    trusted_case,
    AgentProposal(
        proposal_id="program-1",
        generator="solver-agent/model-revision",
        artifact=artifact,
        raw_output=candidate.source,
    ),
)
```

`SandboxCapabilities` records what an integration claims to enforce. It is not
remote attestation. Before using a backend with untrusted source, audit its
image pinning, syscall and privilege policy, network enforcement, mounts,
environment, secret handling, cgroups or equivalent limits, timeout behavior,
and cleanup. See [ADR-0005](adr/0005-generated-program-isolation.md) for the
threat model and rejected local-execution alternatives.
