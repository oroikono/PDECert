"""Framework-neutral runtime records for verifier-guided PDE agents."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .agents import (
    AGENT_TOOL_VERSION,
    AgentEvaluation,
    AgentProposal,
    AgentTrace,
    SymbolicAgentTool,
    evaluate_agent_proposal,
)
from .core import Status


@dataclass(frozen=True)
class AgentToolCall:
    """One structured verifier call, including rejected payloads."""

    call_id: str
    candidate_fields_json: str
    response: Mapping[str, object]
    evaluation: AgentEvaluation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.candidate_fields_json, str):
            raise TypeError("candidate_fields_json must be a string")
        normalized = dict(self.response)
        if not isinstance(normalized.get("ok"), bool):
            raise ValueError("tool responses must contain a boolean ok field")
        if self.evaluation is not None and not normalized["ok"]:
            raise ValueError("a rejected tool call cannot contain an evaluation")
        object.__setattr__(self, "response", MappingProxyType(normalized))

    @property
    def payload_sha256(self) -> str:
        """Return a stable digest for the exact agent-supplied payload."""

        return hashlib.sha256(self.candidate_fields_json.encode("utf-8")).hexdigest()

    @property
    def status(self) -> Status | None:
        """Return the verifier status when the payload materialized."""

        return self.evaluation.report.status if self.evaluation is not None else None

    def to_dict(self, *, include_payload: bool = False) -> dict[str, object]:
        """Serialize the call without exposing raw model text by default."""

        payload: dict[str, object] = {
            "call_id": self.call_id,
            "candidate_fields_sha256": self.payload_sha256,
            "response": dict(self.response),
        }
        if include_payload:
            payload["candidate_fields_json"] = self.candidate_fields_json
        return payload


@dataclass(frozen=True)
class AgentRun:
    """One model run with every verifier interaction kept in order."""

    run_id: str
    problem_id: str
    generator: str
    tool_calls: tuple[AgentToolCall, ...]
    final_output: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run_id"),
            (self.problem_id, "problem_id"),
            (self.generator, "generator"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if any(not isinstance(call, AgentToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain AgentToolCall objects")
        if not isinstance(self.final_output, str):
            raise TypeError("final_output must be a string")
        normalized = dict(self.metadata)
        if any(not isinstance(key, str) or not key for key in normalized):
            raise ValueError("metadata keys must be non-empty strings")
        if any(not isinstance(value, str) for value in normalized.values()):
            raise TypeError("metadata values must be strings")
        object.__setattr__(self, "metadata", MappingProxyType(normalized))

    @property
    def evaluations(self) -> tuple[AgentEvaluation, ...]:
        """Return materialized proposals, excluding rejected tool payloads."""

        return tuple(call.evaluation for call in self.tool_calls if call.evaluation is not None)

    @property
    def trace(self) -> AgentTrace | None:
        """Return the proposal/repair trace when at least one proposal materialized."""

        evaluations = self.evaluations
        if not evaluations:
            return None
        return AgentTrace(self.run_id, self.problem_id, evaluations)

    @property
    def final_output_sha256(self) -> str:
        """Return a digest without publishing the model's final response."""

        return hashlib.sha256(self.final_output.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_raw_outputs: bool = False) -> dict[str, object]:
        """Serialize provenance, calls, and the optional materialized trace."""

        payload: dict[str, object] = {
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "generator": self.generator,
            "metadata": dict(self.metadata),
            "final_output_sha256": self.final_output_sha256,
            "tool_calls": [
                call.to_dict(include_payload=include_raw_outputs) for call in self.tool_calls
            ],
            "trace": (
                self.trace.to_dict(include_raw_outputs=include_raw_outputs)
                if self.trace is not None
                else None
            ),
        }
        if include_raw_outputs:
            payload["final_output"] = self.final_output
        return payload


@dataclass
class SymbolicAgentSession:
    """Record a provider-neutral verifier-guided symbolic agent session."""

    run_id: str
    problem_id: str
    generator: str
    verifier: SymbolicAgentTool
    metadata: Mapping[str, str] = field(default_factory=dict)
    _calls: list[AgentToolCall] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # AgentRun performs the shared text and metadata validation.
        AgentRun(self.run_id, self.problem_id, self.generator, (), "", self.metadata)
        if not isinstance(self.verifier, SymbolicAgentTool):
            raise TypeError("verifier must be a SymbolicAgentTool")

    def submit(self, candidate_fields_json: str) -> str:
        """Verify one structured proposal and return deterministic JSON feedback."""

        if self._closed:
            raise RuntimeError("the agent session is already closed")
        call_id = f"call-{len(self._calls) + 1}"
        try:
            artifact = self.verifier.materialize(candidate_fields_json)
        except (TypeError, ValueError) as error:
            response: dict[str, object] = {
                "ok": False,
                "tool_version": AGENT_TOOL_VERSION,
                "error": str(error),
            }
            self._calls.append(AgentToolCall(call_id, candidate_fields_json, response))
            return _stable_json(response)

        previous = next(
            (call.evaluation for call in reversed(self._calls) if call.evaluation is not None),
            None,
        )
        proposal = AgentProposal(
            proposal_id=f"proposal-{len(self.evaluations) + 1}",
            generator=self.generator,
            artifact=artifact,
            raw_output=candidate_fields_json,
            parent_proposal_id=(previous.proposal.proposal_id if previous is not None else None),
            metadata=self.metadata,
        )
        evaluation = evaluate_agent_proposal(
            self.verifier.trusted_case,
            proposal,
            tolerance=self.verifier.tolerance,
            samples_per_axis=self.verifier.samples_per_axis,
            symbolic_timeout=self.verifier.symbolic_timeout,
            max_expression_ops=self.verifier.max_expression_ops,
        )
        response = self.verifier.feedback(evaluation.report)
        self._calls.append(AgentToolCall(call_id, candidate_fields_json, response, evaluation))
        return _stable_json(response)

    @property
    def evaluations(self) -> tuple[AgentEvaluation, ...]:
        """Return the materialized evaluations accumulated so far."""

        return tuple(call.evaluation for call in self._calls if call.evaluation is not None)

    def finish(self, final_output: str) -> AgentRun:
        """Close the session and return an immutable run record."""

        if self._closed:
            raise RuntimeError("the agent session is already closed")
        self._closed = True
        return AgentRun(
            self.run_id,
            self.problem_id,
            self.generator,
            tuple(self._calls),
            final_output,
            self.metadata,
        )


@dataclass(frozen=True)
class AgentModelMetrics:
    """Verifier-grounded behavioral metrics for one generator identity."""

    generator: str
    runs: int
    tool_calls: int
    rejected_tool_calls: int
    materialized_proposals: int
    first_proved_runs: int
    final_proved_runs: int
    repaired_to_proved_runs: int
    final_status_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return exact counts and rates with explicit denominators."""

        return {
            "generator": self.generator,
            "runs": self.runs,
            "tool_calls": self.tool_calls,
            "rejected_tool_calls": self.rejected_tool_calls,
            "materialized_proposals": self.materialized_proposals,
            "mean_tool_calls_per_run": self.tool_calls / self.runs,
            "first_proved_runs": self.first_proved_runs,
            "first_proved_rate": self.first_proved_runs / self.runs,
            "final_proved_runs": self.final_proved_runs,
            "final_proved_rate": self.final_proved_runs / self.runs,
            "repaired_to_proved_runs": self.repaired_to_proved_runs,
            "final_status_counts": dict(self.final_status_counts),
        }


@dataclass(frozen=True)
class AgentBenchmarkReport:
    """Cross-model summary that does not reinterpret verifier outcomes as labels."""

    models: tuple[AgentModelMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable report with the metric semantics attached."""

        return {
            "metric_scope": (
                "Verifier-grounded agent behavior; this is not independent ground-truth accuracy."
            ),
            "models": [model.to_dict() for model in self.models],
        }


def summarize_agent_runs(runs: Iterable[AgentRun]) -> AgentBenchmarkReport:
    """Group agent behavior by exact generator identity."""

    grouped: dict[str, list[AgentRun]] = {}
    for run in runs:
        if not isinstance(run, AgentRun):
            raise TypeError("runs must contain AgentRun objects")
        grouped.setdefault(run.generator, []).append(run)

    metrics: list[AgentModelMetrics] = []
    for generator in sorted(grouped):
        items = grouped[generator]
        tool_calls = sum(len(run.tool_calls) for run in items)
        rejected = sum(1 for run in items for call in run.tool_calls if call.evaluation is None)
        evaluations = [run.evaluations for run in items]
        first_proved = sum(
            bool(items_) and items_[0].report.status is Status.PROVED for items_ in evaluations
        )
        final_proved = sum(
            bool(items_) and items_[-1].report.status is Status.PROVED for items_ in evaluations
        )
        repaired = sum(
            len(items_) > 1
            and items_[0].report.status is not Status.PROVED
            and items_[-1].report.status is Status.PROVED
            for items_ in evaluations
        )
        final_counts = Counter(
            items_[-1].report.status.value if items_ else "NO_VALID_PROPOSAL"
            for items_ in evaluations
        )
        metrics.append(
            AgentModelMetrics(
                generator=generator,
                runs=len(items),
                tool_calls=tool_calls,
                rejected_tool_calls=rejected,
                materialized_proposals=sum(len(items_) for items_ in evaluations),
                first_proved_runs=first_proved,
                final_proved_runs=final_proved,
                repaired_to_proved_runs=repaired,
                final_status_counts=MappingProxyType(dict(sorted(final_counts.items()))),
            )
        )
    return AgentBenchmarkReport(tuple(metrics))


def _stable_json(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, sort_keys=True)
