"""Candidate artifacts accepted by PDECert verification backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import sympy as sp


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
