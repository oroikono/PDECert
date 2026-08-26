"""Framework-neutral records and tools for agent-generated PDE candidates."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from .artifacts import CallableCandidate, SolutionArtifact, SymbolicCandidate
from .autodiff import AutodiffProblem
from .core import Report, verify_artifact
from .schema import SchemaError, VerificationCase, case_from_dict, case_to_dict


AGENT_TOOL_VERSION = 1


@dataclass(frozen=True)
class AgentProposal:
    """One materialized candidate proposed by an LLM or scientific agent.

    The raw output remains distinct from the parsed solution artifact. A host
    application, not PDECert, is responsible for materializing model output
    into a supported artifact without silently editing it.
    """

    proposal_id: str
    generator: str
    artifact: SolutionArtifact
    raw_output: str
    parent_proposal_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text(self.proposal_id, "proposal_id")
        _validate_text(self.generator, "generator")
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        if self.parent_proposal_id is not None:
            _validate_text(self.parent_proposal_id, "parent_proposal_id")
            if self.parent_proposal_id == self.proposal_id:
                raise ValueError("a proposal cannot be its own parent")
        if not isinstance(self.artifact, SolutionArtifact):
            raise TypeError("artifact must implement the SolutionArtifact protocol")

        normalized = dict(self.metadata)
        if any(not isinstance(key, str) or not key for key in normalized):
            raise ValueError("metadata keys must be non-empty strings")
        if any(not isinstance(value, str) for value in normalized.values()):
            raise TypeError("metadata values must be strings")
        object.__setattr__(self, "metadata", MappingProxyType(normalized))

    @property
    def raw_output_sha256(self) -> str:
        """Return a stable digest without treating the raw output as a label."""

        return hashlib.sha256(self.raw_output.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_raw_output: bool = False) -> dict[str, object]:
        """Return provenance separately from the parsed artifact."""

        payload: dict[str, object] = {
            "proposal_id": self.proposal_id,
            "generator": self.generator,
            "artifact_kind": self.artifact.kind,
            "field_names": list(self.artifact.field_names),
            "parent_proposal_id": self.parent_proposal_id,
            "metadata": dict(self.metadata),
            "raw_output_sha256": self.raw_output_sha256,
        }
        if include_raw_output:
            payload["raw_output"] = self.raw_output
        return payload


@dataclass(frozen=True)
class AgentEvaluation:
    """Machine evidence for one proposal, kept separate from provenance."""

    proposal: AgentProposal
    report: Report

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, AgentProposal):
            raise TypeError("proposal must be an AgentProposal")
        if not isinstance(self.report, Report):
            raise TypeError("report must be a Report")

    def to_dict(self, *, include_raw_output: bool = False) -> dict[str, object]:
        """Return a JSON-compatible proposal and verification record."""

        return {
            "proposal": self.proposal.to_dict(include_raw_output=include_raw_output),
            "verification": self.report.to_dict(),
        }


@dataclass(frozen=True)
class AgentTrace:
    """An ordered proposal, counterexample, and repair history."""

    run_id: str
    problem_id: str
    evaluations: tuple[AgentEvaluation, ...]

    def __post_init__(self) -> None:
        _validate_text(self.run_id, "run_id")
        _validate_text(self.problem_id, "problem_id")
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        if not self.evaluations:
            raise ValueError("agent traces require at least one evaluation")
        if any(not isinstance(item, AgentEvaluation) for item in self.evaluations):
            raise TypeError("agent trace entries must be AgentEvaluation objects")

        seen: set[str] = set()
        for evaluation in self.evaluations:
            proposal = evaluation.proposal
            if proposal.proposal_id in seen:
                raise ValueError(f"duplicate proposal id in agent trace: {proposal.proposal_id}")
            if proposal.parent_proposal_id is not None and proposal.parent_proposal_id not in seen:
                raise ValueError(
                    f"proposal {proposal.proposal_id!r} references a parent that does not "
                    "precede it in the trace"
                )
            seen.add(proposal.proposal_id)

    def to_dict(self, *, include_raw_outputs: bool = False) -> dict[str, object]:
        """Return the ordered trace, excluding raw model text by default."""

        return {
            "run_id": self.run_id,
            "problem_id": self.problem_id,
            "evaluations": [
                item.to_dict(include_raw_output=include_raw_outputs) for item in self.evaluations
            ],
        }


def evaluate_agent_proposal(
    trusted_problem: VerificationCase | AutodiffProblem,
    proposal: AgentProposal,
    *,
    tolerance: float | None = None,
    samples_per_axis: int = 5,
    symbolic_timeout: float | None = None,
    max_expression_ops: int | None = None,
) -> AgentEvaluation:
    """Evaluate one host-materialized proposal against a trusted problem.

    Symbolic proposals require a versioned case whose constraint sources retain
    references to the declared fields. This allows each new expression to be
    substituted into the trusted PDE instead of accidentally reusing residuals
    materialized for an earlier candidate.
    """

    if isinstance(trusted_problem, VerificationCase) and isinstance(
        proposal.artifact, SymbolicCandidate
    ):
        candidate_case = _materialize_symbolic_candidate(trusted_problem, proposal.artifact)
        report = verify_artifact(
            candidate_case.problem,
            SymbolicCandidate.from_expressions(candidate_case.candidate_fields),
            tolerance=tolerance,
            samples_per_axis=samples_per_axis,
            symbolic_timeout=symbolic_timeout,
            max_expression_ops=max_expression_ops,
        )
    elif isinstance(trusted_problem, AutodiffProblem) and isinstance(
        proposal.artifact, CallableCandidate
    ):
        report = verify_artifact(
            trusted_problem,
            proposal.artifact,
            tolerance=tolerance,
            samples_per_axis=samples_per_axis,
            symbolic_timeout=symbolic_timeout,
            max_expression_ops=max_expression_ops,
        )
    else:
        raise TypeError(
            "unsupported trusted problem/artifact pair for agent proposal: "
            f"{type(trusted_problem).__name__} and {type(proposal.artifact).__name__}"
        )
    return AgentEvaluation(proposal, report)


@dataclass(frozen=True)
class SymbolicAgentTool:
    """Restricted JSON tool for an agent working on one trusted symbolic case.

    The trusted problem stays outside the agent-controlled payload. The agent
    may submit only a JSON object mapping the expected field names to restricted
    expression strings accepted by PDECert's versioned schema.
    """

    trusted_case: VerificationCase
    symbolic_timeout: float = 2.0
    max_expression_ops: int = 10_000
    tolerance: float = 1e-9
    samples_per_axis: int = 5
    max_payload_bytes: int = 100_000

    name: ClassVar[str] = "pdecert_verify_symbolic_candidate"
    description: ClassVar[str] = (
        "Verify symbolic candidate fields against a trusted PDE problem and return "
        "PROVED, REFUTED with a witness, or INCONCLUSIVE."
    )
    input_schema: ClassVar[dict[str, object]] = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    def __post_init__(self) -> None:
        if not isinstance(self.trusted_case, VerificationCase):
            raise TypeError("trusted_case must be a VerificationCase")
        _validate_symbolic_template(self.trusted_case)
        if self.max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be positive")

    def evaluate(self, candidate_fields_json: str) -> dict[str, object]:
        """Return stable tool feedback without executing generated code."""

        if not isinstance(candidate_fields_json, str):
            return _tool_error("candidate fields must be supplied as a JSON string")
        if len(candidate_fields_json.encode("utf-8")) > self.max_payload_bytes:
            return _tool_error(f"candidate payload exceeds the {self.max_payload_bytes}-byte limit")
        try:
            candidate_fields = json.loads(candidate_fields_json)
        except json.JSONDecodeError as error:
            return _tool_error(f"invalid JSON: {error.msg}")
        if not isinstance(candidate_fields, dict):
            return _tool_error("candidate payload must be a JSON object")
        if set(candidate_fields) != set(self.trusted_case.field_names):
            expected = ", ".join(self.trusted_case.field_names)
            return _tool_error(f"candidate fields must be exactly: {expected}")

        try:
            ordered_fields = {
                name: candidate_fields[name] for name in self.trusted_case.field_names
            }
            candidate_case = _materialize_symbolic_fields(self.trusted_case, ordered_fields)
        except SchemaError as error:
            return _tool_error(str(error))

        report = verify_artifact(
            candidate_case.problem,
            SymbolicCandidate.from_expressions(candidate_case.candidate_fields),
            tolerance=self.tolerance,
            samples_per_axis=self.samples_per_axis,
            symbolic_timeout=self.symbolic_timeout,
            max_expression_ops=self.max_expression_ops,
        )
        return {
            "ok": True,
            "tool_version": AGENT_TOOL_VERSION,
            "problem": self.trusted_case.problem.name,
            "field_names": list(self.trusted_case.field_names),
            "report": report.to_dict(),
        }

    def __call__(self, candidate_fields_json: str) -> str:
        """Return deterministic JSON suitable for an agent framework tool."""

        return json.dumps(self.evaluate(candidate_fields_json), sort_keys=True)


def _tool_error(message: str) -> dict[str, object]:
    return {
        "ok": False,
        "tool_version": AGENT_TOOL_VERSION,
        "error": message,
    }


def _materialize_symbolic_candidate(
    trusted_case: VerificationCase,
    artifact: SymbolicCandidate,
) -> VerificationCase:
    _validate_symbolic_template(trusted_case)
    if artifact.field_names != trusted_case.field_names:
        expected = ", ".join(trusted_case.field_names)
        raise ValueError(f"symbolic proposal fields must be exactly: {expected}")
    return _materialize_symbolic_fields(
        trusted_case,
        {name: str(expression) for name, expression in artifact.fields},
    )


def _materialize_symbolic_fields(
    trusted_case: VerificationCase,
    rendered_fields: Mapping[str, object],
) -> VerificationCase:
    if any(not isinstance(expression, str) for expression in rendered_fields.values()):
        raise SchemaError("$.fields: candidate expressions must be strings")
    payload = case_to_dict(trusted_case)
    payload["fields"] = dict(rendered_fields)
    return case_from_dict(payload)


def _validate_symbolic_template(trusted_case: VerificationCase) -> None:
    sources = tuple(
        constraint.source
        for constraint in trusted_case.problem.pde_residuals + trusted_case.problem.conditions
        if constraint.source is not None
    )
    missing = [
        name
        for name in trusted_case.field_names
        if not any(_contains_identifier(source, name) for source in sources)
    ]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            "trusted symbolic cases must retain field-referenced constraint sources; "
            f"missing: {names}"
        )


def _contains_identifier(source: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source) is not None


def _validate_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
