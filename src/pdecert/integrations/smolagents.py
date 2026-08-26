"""Optional smolagents adapter for structured PDE verification calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..agent_runtime import AgentRun, SymbolicAgentSession
from ..agents import SymbolicAgentTool
from ..schema import VerificationCase


class SmolagentsUnavailable(ImportError):
    """Raised when the optional smolagents dependency is not installed."""


def build_smolagents_tool(session: SymbolicAgentSession) -> object:
    """Return a real ``smolagents.Tool`` backed by a recorded PDECert session."""

    if not isinstance(session, SymbolicAgentSession):
        raise TypeError("session must be a SymbolicAgentSession")
    Tool, _ = _smolagents_classes()

    class PDECertSymbolicTool(Tool):
        name = SymbolicAgentTool.name
        description = SymbolicAgentTool.description
        inputs = {
            "candidate_fields_json": {
                "type": "string",
                "description": (
                    "A JSON object mapping every expected PDE field name to one restricted "
                    "symbolic expression string."
                ),
            }
        }
        output_type = "string"

        def forward(self, candidate_fields_json: str) -> str:
            return session.submit(candidate_fields_json)

    return PDECertSymbolicTool()


def run_smolagents_symbolic_agent(
    *,
    trusted_case: VerificationCase,
    model: object,
    prompt: str,
    run_id: str,
    problem_id: str,
    generator: str,
    max_steps: int = 6,
    metadata: Mapping[str, str] | None = None,
    verifier_options: Mapping[str, object] | None = None,
    agent_options: Mapping[str, object] | None = None,
) -> AgentRun:
    """Run a structured smolagents loop and return its complete verifier trace.

    ``model`` is an initialized smolagents model, such as
    ``InferenceClientModel`` or ``LiteLLMModel``. Calling ``agent.run`` performs
    the provider API calls configured by that model. PDECert neither reads nor
    stores provider credentials.

    Only ``ToolCallingAgent`` is used here. No model-written Python is executed.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    _, ToolCallingAgent = _smolagents_classes()
    verifier = SymbolicAgentTool(trusted_case, **dict(verifier_options or {}))
    session = SymbolicAgentSession(
        run_id=run_id,
        problem_id=problem_id,
        generator=generator,
        verifier=verifier,
        metadata=dict(metadata or {}),
    )
    tool = build_smolagents_tool(session)

    options: dict[str, Any] = dict(agent_options or {})
    forbidden = {"model", "tools", "max_steps"} & set(options)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ValueError(f"agent_options cannot override: {names}")
    agent = ToolCallingAgent(tools=[tool], model=model, max_steps=max_steps, **options)
    task = (
        f"{prompt.strip()}\n\n"
        "Submit candidate fields to pdecert_verify_symbolic_candidate. Use its exact "
        "REFUTED witness or INCONCLUSIVE reason to revise the candidate when needed. "
        "Do not claim success unless the tool returns PROVED."
    )
    final_output = agent.run(task)
    return session.finish(str(final_output))


def _smolagents_classes() -> tuple[type[object], type[object]]:
    try:
        from smolagents import Tool, ToolCallingAgent
    except ImportError as error:
        raise SmolagentsUnavailable(
            "install PDECert with the 'agents' extra to use the smolagents adapter"
        ) from error
    return Tool, ToolCallingAgent
