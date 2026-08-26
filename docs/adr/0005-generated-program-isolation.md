# ADR-0005: Deny-by-default generated-program isolation

## Status

Accepted

## Date

2026-08-26

## Deciders

PDECert maintainers

## Context

Some solvers and language-model agents emit programs rather than a closed-form
expression or an already loaded differentiable function. Evaluating that output
would let PDECert cover a useful artifact class, but candidate source is fully
untrusted. It can read credentials, alter files, contact the network, fork
processes, consume unbounded resources, or exploit the interpreter and host
kernel.

Python restrictions, an import allowlist, or a child process under the same user
do not create a security boundary. The smolagents project likewise documents
that its local Python executor is not a security sandbox. PDECert must not imply
that syntactic filtering or `subprocess` alone makes generated code safe.

The first program-output contract is intentionally narrow: Python source emits
one bounded JSON object mapping declared field names to symbolic expression
strings. Those strings still pass through PDECert's restricted parser and
ordinary verifier. Successful program termination is provenance evidence, not a
solution certificate.

## Decision

PDECert represents source as a non-executing `ProgramCandidate` and refuses to
run it by default. Execution is possible only through an explicitly supplied
`ProgramSandbox` implementation that declares all of these capabilities:

- process isolation from the host;
- disabled or isolated networking;
- an ephemeral filesystem;
- read-only candidate source;
- enforced wall-time, CPU, memory, process, and output limits;
- isolation from host and provider secrets.

The core has no `exec`, shell, local-interpreter, or subprocess fallback. It
rejects a backend before execution when any required capability is absent. The
backend receives exact limits and must enforce them without invoking a shell.
PDECert then independently checks the returned stream sizes, timeout and exit
state, JSON shape, exact field names, and string value types.

Backend capability declarations are configuration claims, not remote
attestation. Deployers remain responsible for choosing and auditing a real
container, microVM, WebAssembly, or managed sandbox implementation. A production
backend should pin its runtime image and record its digest and policy version in
benchmark provenance.

## Options considered

### Option A: Execute in the PDECert Python process

| Aspect | Assessment |
|---|---|
| Isolation | None |
| Portability | High |
| Security | Unacceptable for untrusted source |
| Operational cost | Low |

Rejected because a parser or restricted global namespace cannot defend the host
against arbitrary Python behavior and implementation escapes.

### Option B: Run a local subprocess with timeouts and resource limits

| Aspect | Assessment |
|---|---|
| Isolation | Shares user, filesystem, kernel, environment, and usually network |
| Portability | Resource controls vary by operating system |
| Security | Useful containment, not a sufficient trust boundary |
| Operational cost | Low |

Rejected as the default security boundary. A subprocess may be useful for
trusted development code, but PDECert will not label it safe for generated
programs.

### Option C: Require an explicit isolated backend and deny by default

| Aspect | Assessment |
|---|---|
| Isolation | Supplied by an audited external sandbox |
| Portability | Backend-specific, core contract is portable |
| Security | Fail-closed with explicit required capabilities |
| Operational cost | Higher setup cost |

Accepted because it keeps security claims honest while allowing Docker,
microVM, WebAssembly, and managed sandbox integrations to evolve independently.

### Option D: Never support generated programs

| Aspect | Assessment |
|---|---|
| Isolation | No execution risk |
| Coverage | Excludes an important solver and agent artifact class |
| Extensibility | No integration path |
| Operational cost | None |

Rejected because a narrow, explicit adapter contract enables the use case
without weakening the default.

## Trade-off analysis

The decision prioritizes a defensible trust boundary over one-command local
execution. Users cannot run a `ProgramCandidate` until they configure a backend,
and PDECert does not yet ship a production backend. In return, importing or
constructing a candidate is inert, missing isolation fails before any backend
call, output has a deterministic contract, and future backends can be compared
against the same tests.

This boundary does not certify the sandbox itself and cannot eliminate kernel,
container-runtime, or managed-service vulnerabilities. It limits what PDECert
claims and makes the remaining trust assumption visible.

## Consequences

### Positive

- Generated source is inert unless an isolation backend is explicitly passed.
- No convenience path can silently fall back to local execution.
- Program output rejoins the existing restricted symbolic and verification
  pipeline rather than becoming a parallel trust path.
- Benchmark records can bind source, sandbox, and output through stable digests.

### Negative

- Local examples use a fake backend contract and do not demonstrate real
  untrusted execution.
- Production use requires additional infrastructure and backend-specific tests.
- Self-declared capabilities require deployment review; they are not proof of
  isolation.
- The first output contract supports symbolic fields only, not checkpoints,
  meshes, or arbitrary files.

## Action items

- Add conformance tests reusable by third-party sandbox backends.
- Define provenance fields for runtime image digest, policy version, and backend
  revision before publishing program-generated benchmark runs.
- Implement one audited optional backend without changing the disabled default.
- Extend the output contract only through a separate ADR for each artifact type.
- Add adversarial escape and resource-exhaustion tests to every production
  backend's own CI environment.
