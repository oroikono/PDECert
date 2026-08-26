import json
import unittest

from pdecert import (
    PROGRAM_SOURCE_MAX_BYTES,
    ProgramCandidate,
    ProgramExecutionError,
    ProgramIsolationError,
    ProgramLimits,
    SandboxCapabilities,
    SandboxResult,
    SymbolicAgentTool,
    case_from_dict,
    execute_program_candidate,
)


def heat_case():
    return case_from_dict(
        {
            "schema_version": 3,
            "name": "heat equation",
            "variables": ["x", "t"],
            "domains": {"x": [0.0, 1.0], "t": [0.0, 1.0]},
            "parameters": {},
            "pde_residuals": [{"name": "heat PDE", "expression": "D(u, t) - D(u, x, 2)"}],
            "conditions": [
                {"name": "initial condition", "expression": "At(u, t, 0) - sin(pi*x)"},
                {"name": "left boundary", "expression": "At(u, x, 0)"},
                {"name": "right boundary", "expression": "At(u, x, 1)"},
            ],
            "fields": {"u": "exp(-pi**2*t)*sin(pi*x)"},
        }
    )


SAFE_CAPABILITIES = SandboxCapabilities(
    process_isolation=True,
    network_isolation=True,
    ephemeral_filesystem=True,
    read_only_source=True,
    resource_limits=True,
    secret_isolation=True,
)


class FakeSandbox:
    name = "test-isolated-backend"
    capabilities = SAFE_CAPABILITIES

    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, candidate, limits):
        self.calls.append((candidate, limits))
        return self.result


class ProgramCandidateTests(unittest.TestCase):
    def test_constructing_candidate_is_inert_and_content_addressed(self):
        candidate = ProgramCandidate("raise RuntimeError('must not run')", ("u",))
        self.assertEqual(candidate.kind, "program")
        self.assertEqual(candidate.field_names, ("u",))
        self.assertEqual(len(candidate.source_sha256), 64)

    def test_candidate_rejects_unsupported_or_oversized_source(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            ProgramCandidate("print('{}')", ("u",), language="javascript")
        with self.assertRaisesRegex(ValueError, "artifact limit"):
            ProgramCandidate("x" * (PROGRAM_SOURCE_MAX_BYTES + 1), ("u",))
        with self.assertRaisesRegex(ValueError, "identifiers"):
            ProgramCandidate("print('{}')", ("u[0]",))
        with self.assertRaisesRegex(TypeError, "iterable"):
            ProgramCandidate("print('{}')", "u")

    def test_default_fails_closed_before_source_execution(self):
        candidate = ProgramCandidate("raise RuntimeError('must not run')", ("u",))
        with self.assertRaisesRegex(ProgramIsolationError, "missing required"):
            execute_program_candidate(candidate)

    def test_incomplete_backend_is_rejected_before_execute(self):
        class InsecureSandbox(FakeSandbox):
            capabilities = SandboxCapabilities(process_isolation=True)

        backend = InsecureSandbox(SandboxResult(0, '{"u": "0"}', "", 0.1))
        with self.assertRaisesRegex(ProgramIsolationError, "network_isolation"):
            execute_program_candidate(ProgramCandidate("print('{}')", ("u",)), backend)
        self.assertEqual(backend.calls, [])

    def test_isolated_result_is_bounded_and_materializes_through_restricted_parser(self):
        candidate = ProgramCandidate("generated source", ("u",))
        expression = "exp(-pi**2*t)*sin(pi*x)"
        backend = FakeSandbox(SandboxResult(0, json.dumps({"u": expression}), "", 0.2))

        output = execute_program_candidate(candidate, backend)
        artifact = output.materialize(SymbolicAgentTool(heat_case()))

        self.assertEqual(output.fields, {"u": expression})
        self.assertEqual(output.program_sha256, candidate.source_sha256)
        self.assertEqual(artifact.field_names, ("u",))
        self.assertEqual(len(backend.calls), 1)
        self.assertIsInstance(backend.calls[0][1], ProgramLimits)

    def test_nonzero_timeout_and_invalid_output_never_materialize(self):
        candidate = ProgramCandidate("generated source", ("u",))
        cases = (
            (SandboxResult(2, "{}", "error", 0.1), "exited with code 2"),
            (SandboxResult(0, "{}", "", 10.0, timed_out=True), "time limit"),
            (SandboxResult(0, "not-json", "", 0.1), "not valid JSON"),
            (SandboxResult(0, '{"v": "0"}', "", 0.1), "fields must be exactly"),
            (SandboxResult(0, '{"u": 0}', "", 0.1), "expressions must be strings"),
            (SandboxResult(0, '{"u": "0", "u": "1"}', "", 0.1), "duplicate field"),
            (SandboxResult(0, '{"u": "0"}', "", 11.0), "wall-time limit"),
        )
        for result, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ProgramExecutionError, message):
                    execute_program_candidate(candidate, FakeSandbox(result))

    def test_core_rechecks_backend_output_limits(self):
        candidate = ProgramCandidate("generated source", ("u",))
        limits = ProgramLimits(max_stdout_bytes=4)
        backend = FakeSandbox(SandboxResult(0, '{"u": "0"}', "", 0.1))
        with self.assertRaisesRegex(ProgramExecutionError, "stdout exceeded"):
            execute_program_candidate(candidate, backend, limits=limits)


if __name__ == "__main__":
    unittest.main()
