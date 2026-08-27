"""Deny-by-default execution contract for generated solver programs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .agents import SymbolicAgentTool
from .artifacts import ProgramCandidate, SymbolicCandidate


class ProgramError(Exception):
    """Base class for generated-program boundary errors."""


class ProgramIsolationError(ProgramError):
    """Raised before execution when a backend lacks required isolation."""


class ProgramExecutionError(ProgramError):
    """Raised when an isolated execution does not produce a valid artifact."""


@dataclass(frozen=True)
class ProgramLimits:
    """Resource ceilings every configured sandbox must enforce."""

    wall_time_seconds: float = 10.0
    cpu_time_seconds: float = 5.0
    memory_bytes: int = 512 * 1024 * 1024
    max_processes: int = 16
    max_stdout_bytes: int = 100_000
    max_stderr_bytes: int = 100_000

    def __post_init__(self) -> None:
        for value, label in (
            (self.wall_time_seconds, "wall_time_seconds"),
            (self.cpu_time_seconds, "cpu_time_seconds"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive finite number")
        for value, label in (
            (self.memory_bytes, "memory_bytes"),
            (self.max_processes, "max_processes"),
            (self.max_stdout_bytes, "max_stdout_bytes"),
            (self.max_stderr_bytes, "max_stderr_bytes"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class SandboxCapabilities:
    """Isolation properties a backend explicitly claims to enforce."""

    process_isolation: bool = False
    network_isolation: bool = False
    ephemeral_filesystem: bool = False
    read_only_source: bool = False
    resource_limits: bool = False
    secret_isolation: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(getattr(self, name), bool)
            for name in (
                "process_isolation",
                "network_isolation",
                "ephemeral_filesystem",
                "read_only_source",
                "resource_limits",
                "secret_isolation",
            )
        ):
            raise TypeError("sandbox capabilities must be booleans")

    def missing_required(self) -> tuple[str, ...]:
        """Return required capabilities that are not enabled."""

        return tuple(
            name
            for name in (
                "process_isolation",
                "network_isolation",
                "ephemeral_filesystem",
                "read_only_source",
                "resource_limits",
                "secret_isolation",
            )
            if not getattr(self, name)
        )


@dataclass(frozen=True)
class SandboxResult:
    """Raw bounded result returned by an external isolation backend."""

    exit_code: int
    stdout: str
    stderr: str
    wall_time_seconds: float
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("sandbox output streams must be strings")
        if (
            not isinstance(self.wall_time_seconds, (int, float))
            or isinstance(self.wall_time_seconds, bool)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise ValueError("wall_time_seconds must be a finite nonnegative number")
        if not isinstance(self.timed_out, bool):
            raise TypeError("timed_out must be a boolean")


@runtime_checkable
class ProgramSandbox(Protocol):
    """External executor contract; implementations provide the isolation."""

    name: str
    capabilities: SandboxCapabilities

    def execute(self, candidate: ProgramCandidate, limits: ProgramLimits) -> SandboxResult:
        """Execute without a shell while enforcing every supplied limit."""


@dataclass(frozen=True)
class ProgramOutput:
    """Validated symbolic fields emitted by one isolated program run."""

    fields: Mapping[str, str]
    program_sha256: str
    sandbox: str
    wall_time_seconds: float

    def __post_init__(self) -> None:
        normalized = dict(self.fields)
        if (
            not normalized
            or any(not isinstance(key, str) or not key for key in normalized)
            or any(not isinstance(value, str) for value in normalized.values())
        ):
            raise ValueError("program output fields must be a non-empty string mapping")
        if not isinstance(self.program_sha256, str) or len(self.program_sha256) != 64:
            raise ValueError("program_sha256 must be a SHA-256 hex digest")
        try:
            int(self.program_sha256, 16)
        except ValueError as error:
            raise ValueError("program_sha256 must be a SHA-256 hex digest") from error
        if not isinstance(self.sandbox, str) or not self.sandbox.strip():
            raise ValueError("sandbox must be a non-empty string")
        if (
            not isinstance(self.wall_time_seconds, (int, float))
            or isinstance(self.wall_time_seconds, bool)
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise ValueError("wall_time_seconds must be a finite nonnegative number")
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    def materialize(self, verifier: SymbolicAgentTool) -> SymbolicCandidate:
        """Parse emitted fields with the same restricted symbolic grammar as agents."""

        if not isinstance(verifier, SymbolicAgentTool):
            raise TypeError("verifier must be a SymbolicAgentTool")
        return verifier.materialize(json.dumps(dict(self.fields), sort_keys=True))

    def to_dict(self) -> dict[str, object]:
        """Return execution evidence without including the source program."""

        return {
            "fields": dict(self.fields),
            "program_sha256": self.program_sha256,
            "sandbox": self.sandbox,
            "wall_time_seconds": self.wall_time_seconds,
        }


class DisabledProgramSandbox:
    """Explicit default that can never execute candidate source."""

    name = "disabled"
    capabilities = SandboxCapabilities()

    def execute(self, candidate: ProgramCandidate, limits: ProgramLimits) -> SandboxResult:
        del candidate, limits
        raise ProgramIsolationError("generated-program execution is disabled")


def execute_program_candidate(
    candidate: ProgramCandidate,
    sandbox: ProgramSandbox | None = None,
    *,
    limits: ProgramLimits | None = None,
) -> ProgramOutput:
    """Execute only through a capability-complete external sandbox backend.

    PDECert does not contain an in-process, subprocess, or shell fallback. The
    backend is responsible for enforcing its declared isolation properties and
    the exact resource limits. PDECert then checks termination and validates a
    bounded JSON output contract before it can become a symbolic artifact.
    """

    if not isinstance(candidate, ProgramCandidate):
        raise TypeError("candidate must be a ProgramCandidate")
    configured = sandbox if sandbox is not None else DisabledProgramSandbox()
    if not isinstance(configured, ProgramSandbox):
        raise TypeError("sandbox must implement the ProgramSandbox protocol")
    if not isinstance(configured.name, str) or not configured.name.strip():
        raise ProgramIsolationError("sandbox name must be a non-empty string")
    if not isinstance(configured.capabilities, SandboxCapabilities):
        raise ProgramIsolationError("sandbox capabilities must use SandboxCapabilities")
    missing = configured.capabilities.missing_required()
    if missing:
        raise ProgramIsolationError(
            "sandbox is missing required isolation capabilities: " + ", ".join(missing)
        )

    policy = limits if limits is not None else ProgramLimits()
    if not isinstance(policy, ProgramLimits):
        raise TypeError("limits must be ProgramLimits")
    result = configured.execute(candidate, policy)
    if not isinstance(result, SandboxResult):
        raise ProgramExecutionError("sandbox returned an unsupported result type")
    if len(result.stdout.encode("utf-8")) > policy.max_stdout_bytes:
        raise ProgramExecutionError("sandbox stdout exceeded the configured byte limit")
    if len(result.stderr.encode("utf-8")) > policy.max_stderr_bytes:
        raise ProgramExecutionError("sandbox stderr exceeded the configured byte limit")
    if result.timed_out:
        raise ProgramExecutionError("sandbox execution exceeded its time limit")
    if result.wall_time_seconds > policy.wall_time_seconds:
        raise ProgramExecutionError("sandbox result exceeded the configured wall-time limit")
    if result.exit_code != 0:
        raise ProgramExecutionError(f"sandbox execution exited with code {result.exit_code}")

    try:
        fields = json.loads(result.stdout, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise ProgramExecutionError(f"sandbox stdout is not valid JSON: {error.msg}") from error
    if not isinstance(fields, dict):
        raise ProgramExecutionError("sandbox stdout must be one JSON object")
    if set(fields) != set(candidate.field_names):
        expected = ", ".join(candidate.field_names)
        raise ProgramExecutionError(f"program output fields must be exactly: {expected}")
    if any(not isinstance(value, str) for value in fields.values()):
        raise ProgramExecutionError("program output expressions must be strings")

    ordered = {name: fields[name] for name in candidate.field_names}
    return ProgramOutput(
        fields=ordered,
        program_sha256=candidate.source_sha256,
        sandbox=configured.name,
        wall_time_seconds=result.wall_time_seconds,
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProgramExecutionError(f"sandbox stdout contains duplicate field: {key}")
        result[key] = value
    return result
