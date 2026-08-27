"""Candidate artifacts accepted by PDECert verification backends."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import sympy as sp


PROGRAM_SOURCE_MAX_BYTES = 1_000_000


@runtime_checkable
class SolutionArtifact(Protocol):
    """Minimal public contract shared by candidate solution representations."""

    kind: str

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return candidate field names in stable order."""


@dataclass(frozen=True)
class SymbolicCandidate:
    """One or more named SymPy fields representing an analytical candidate."""

    fields: tuple[tuple[str, sp.Expr], ...]
    kind: ClassVar[str] = "symbolic"

    def __post_init__(self) -> None:
        _validate_fields(self.fields, value_type=sp.Expr, value_label="SymPy expressions")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return symbolic field names in stable order."""

        return tuple(name for name, _ in self.fields)

    @classmethod
    def from_expressions(
        cls,
        expressions: Iterable[sp.Expr] | Mapping[str, sp.Expr],
    ) -> SymbolicCandidate:
        """Build a named symbolic artifact from the legacy verifier input."""

        if isinstance(expressions, Mapping):
            fields = tuple(expressions.items())
        else:
            fields = tuple(
                (f"candidate[{index}]", expression) for index, expression in enumerate(expressions)
            )
        return cls(fields)


@dataclass(frozen=True)
class CallableCandidate:
    """Named differentiable fields evaluated by an optional autodiff backend.

    Each callable receives one tensor of shape ``(points, variables)`` and must
    return either ``(points,)`` or ``(points, 1)``. Fields must evaluate points
    independently: cross-sample attention and training-mode batch operations do
    not have pointwise PDE derivative semantics. The initial backend is PyTorch;
    storing the backend explicitly keeps future JAX support additive.
    """

    fields: tuple[tuple[str, Callable[[object], object]], ...]
    backend: str = "torch"
    dtype: str | None = None
    device: str | None = None
    kind: ClassVar[str] = "callable"

    def __post_init__(self) -> None:
        _validate_fields(self.fields, value_label="callables")
        if any(not callable(field) for _, field in self.fields):
            raise TypeError("callable candidate fields must be callable")
        if self.backend != "torch":
            raise ValueError("the only supported callable backend is 'torch'")
        if self.dtype not in {None, "float32", "float64"}:
            raise ValueError("callable candidate dtype must be float32, float64, or None")
        if self.device is not None and not self.device.strip():
            raise ValueError("callable candidate device must be a non-empty string")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return callable field names in stable order."""

        return tuple(name for name, _ in self.fields)

    @classmethod
    def from_mapping(
        cls,
        fields: Mapping[str, Callable[[object], object]],
        *,
        backend: str = "torch",
        dtype: str | None = None,
        device: str | None = None,
    ) -> CallableCandidate:
        """Build a callable artifact while preserving mapping order."""

        return cls(tuple(fields.items()), backend=backend, dtype=dtype, device=device)


@dataclass(frozen=True)
class ProgramCandidate:
    """Untrusted source that may produce named symbolic candidate fields.

    Constructing this record never executes ``source``. Execution is available
    only through :func:`pdecert.execute_program_candidate`, which requires an
    explicitly configured isolation backend satisfying PDECert's sandbox
    contract. A successful program writes one JSON object mapping the declared
    field names to restricted symbolic expression strings.
    """

    source: str
    declared_field_names: tuple[str, ...]
    language: str = "python"
    kind: ClassVar[str] = "program"

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("program source must be a non-empty string")
        if len(self.source.encode("utf-8")) > PROGRAM_SOURCE_MAX_BYTES:
            raise ValueError(
                f"program source exceeds the {PROGRAM_SOURCE_MAX_BYTES}-byte artifact limit"
            )
        if isinstance(self.declared_field_names, str):
            raise TypeError("declared_field_names must be an iterable of field names")
        names = tuple(self.declared_field_names)
        _validate_program_field_names(names)
        object.__setattr__(self, "declared_field_names", names)
        if self.language != "python":
            raise ValueError("the only supported program language is 'python'")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the fields the program is required to emit."""

        return self.declared_field_names

    @property
    def source_sha256(self) -> str:
        """Return a stable digest for provenance and sandbox requests."""

        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _validate_fields(
    fields: tuple[tuple[str, object], ...],
    *,
    value_type: type[object] | None = None,
    value_label: str,
) -> None:
    if not fields:
        raise ValueError("at least one candidate field is required")
    names = tuple(name for name, _ in fields)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate field names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("candidate field names must be unique")
    if value_type is not None and any(not isinstance(value, value_type) for _, value in fields):
        raise TypeError(f"candidate fields must be {value_label}")


def _validate_program_field_names(names: tuple[str, ...]) -> None:
    if not names:
        raise ValueError("at least one candidate field is required")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate field names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("candidate field names must be unique")
    if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names):
        raise ValueError("candidate field names must be identifiers")
