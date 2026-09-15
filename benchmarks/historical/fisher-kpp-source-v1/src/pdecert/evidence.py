"""Typed evidence carried by conservative PDE verification reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class EvidenceLevel(str, Enum):
    """Mathematical strength supporting a decisive verification outcome."""

    EXACT = "EXACT"
    RIGOROUS_BOUND = "RIGOROUS_BOUND"
    EMPIRICAL = "EMPIRICAL"


class EvidenceKind(str, Enum):
    """Stable category for one item of obligation-level evidence."""

    EXACT_CERTIFICATE = "EXACT_CERTIFICATE"
    RIGOROUS_BOUND = "RIGOROUS_BOUND"
    EMPIRICAL_COUNTEREXAMPLE = "EMPIRICAL_COUNTEREXAMPLE"
    EMPIRICAL_PASS = "EMPIRICAL_PASS"
    ABSTENTION = "ABSTENTION"


class EvidenceOutcome(str, Enum):
    """What one evidence event establishes about its named obligation."""

    DISCHARGED = "DISCHARGED"
    REFUTED = "REFUTED"
    OBSERVED_PASS = "OBSERVED_PASS"
    ABSTAINED = "ABSTAINED"


class BoundType(str, Enum):
    """Mathematical quantity controlled by a rigorous bound."""

    UNIFORM_RESIDUAL = "UNIFORM_RESIDUAL"
    BOUNDARY_TRACE = "BOUNDARY_TRACE"
    SOLUTION_ERROR = "SOLUTION_ERROR"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Witness:
    """A concrete reason why a candidate was refuted."""

    constraint: str
    point: dict[str, float | str]
    residual: float | str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, str) or not self.constraint.strip():
            raise ValueError("witness constraint must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("witness reason must be a non-empty string")
        for name, value in self.point.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("witness point names must be non-empty strings")
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError("witness point strings must be non-empty")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("witness point values must be finite numbers or strings")
        if isinstance(self.residual, str):
            if not self.residual.strip():
                raise ValueError("witness residual strings must be non-empty")
        elif isinstance(self.residual, bool) or not isinstance(self.residual, (int, float)):
            raise ValueError("witness residual must be a number or string")
        elif math.isnan(self.residual):
            raise ValueError("witness residual cannot be NaN")

    def to_dict(self) -> dict[str, object]:
        """Return strict-JSON-safe witness data."""

        residual: float | str = self.residual
        if isinstance(residual, float) and not math.isfinite(residual):
            residual = "infinity" if residual > 0 else "-infinity"
        return {
            "constraint": self.constraint,
            "point": dict(self.point),
            "residual": residual,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BoundEvidence:
    """Machine-readable scope of one rigorous numerical bound.

    A residual or boundary bound is not a solution-error guarantee. ``bound_type``
    preserves that distinction, while ``assumptions`` and ``constants`` expose
    the hypotheses required to reproduce the claim.
    """

    bound_type: BoundType
    quantity: str
    upper_bound: float
    norm: str
    scope: str
    assumptions: tuple[str, ...] = ()
    constants: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bound_type, BoundType):
            raise ValueError("bound_type must be a BoundType")
        for name, value in {
            "quantity": self.quantity,
            "norm": self.norm,
            "scope": self.scope,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"bound {name} must be a non-empty string")
        if not math.isfinite(self.upper_bound) or self.upper_bound < 0:
            raise ValueError("bound upper_bound must be finite and nonnegative")
        if any(not isinstance(item, str) or not item.strip() for item in self.assumptions):
            raise ValueError("bound assumptions must be non-empty strings")
        for name, value in self.constants.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("bound constant names must be non-empty strings")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("bound constants must be finite numbers or strings")
            if not isinstance(value, (int, float, str)) or isinstance(value, bool):
                raise ValueError("bound constants must be finite numbers or strings")
        object.__setattr__(self, "constants", MappingProxyType(dict(self.constants)))

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation of this bound."""

        return {
            "bound_type": self.bound_type.value,
            "quantity": self.quantity,
            "upper_bound": self.upper_bound,
            "norm": self.norm,
            "scope": self.scope,
            "assumptions": list(self.assumptions),
            "constants": dict(self.constants),
        }


@dataclass(frozen=True)
class EvidenceEvent:
    """Evidence emitted by one checker for one stable obligation identifier."""

    obligation_id: str
    checker: str
    kind: EvidenceKind
    outcome: EvidenceOutcome
    level: EvidenceLevel | None
    detail: str
    witness: Witness | None = None
    bound: BoundEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise ValueError("evidence kind must be an EvidenceKind")
        if not isinstance(self.outcome, EvidenceOutcome):
            raise ValueError("evidence outcome must be an EvidenceOutcome")
        if self.level is not None and not isinstance(self.level, EvidenceLevel):
            raise ValueError("evidence level must be an EvidenceLevel or None")
        for name, value in {
            "obligation_id": self.obligation_id,
            "checker": self.checker,
            "detail": self.detail,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence {name} must be a non-empty string")

        if self.kind is EvidenceKind.EXACT_CERTIFICATE:
            if self.level is not EvidenceLevel.EXACT:
                raise ValueError("exact certificates require EXACT evidence")
            if self.outcome not in {EvidenceOutcome.DISCHARGED, EvidenceOutcome.REFUTED}:
                raise ValueError("exact certificates must discharge or refute an obligation")
        elif self.kind is EvidenceKind.RIGOROUS_BOUND:
            if self.level is not EvidenceLevel.RIGOROUS_BOUND:
                raise ValueError("rigorous bounds require RIGOROUS_BOUND evidence")
            if self.outcome is not EvidenceOutcome.DISCHARGED:
                raise ValueError("version 1 rigorous bounds can only discharge obligations")
            if self.bound is None:
                raise ValueError("rigorous-bound evidence requires a bound payload")
        elif self.kind is EvidenceKind.EMPIRICAL_COUNTEREXAMPLE:
            if self.level is not EvidenceLevel.EMPIRICAL:
                raise ValueError("empirical counterexamples require EMPIRICAL evidence")
            if self.outcome is not EvidenceOutcome.REFUTED:
                raise ValueError("empirical counterexamples must refute an obligation")
        elif self.kind is EvidenceKind.EMPIRICAL_PASS:
            if self.level is not EvidenceLevel.EMPIRICAL:
                raise ValueError("empirical passes require EMPIRICAL evidence")
            if self.outcome is not EvidenceOutcome.OBSERVED_PASS:
                raise ValueError("empirical passes must use OBSERVED_PASS")
        elif self.kind is EvidenceKind.ABSTENTION:
            if self.level is not None or self.outcome is not EvidenceOutcome.ABSTAINED:
                raise ValueError("abstentions require no level and the ABSTAINED outcome")

        if self.outcome is EvidenceOutcome.REFUTED and self.witness is None:
            raise ValueError("refuting evidence requires a witness")
        if self.outcome is not EvidenceOutcome.REFUTED and self.witness is not None:
            raise ValueError("only refuting evidence may carry a witness")
        if self.kind is not EvidenceKind.RIGOROUS_BOUND and self.bound is not None:
            raise ValueError("only rigorous-bound evidence may carry a bound payload")

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation of this evidence event."""

        return {
            "obligation_id": self.obligation_id,
            "checker": self.checker,
            "kind": self.kind.value,
            "outcome": self.outcome.value,
            "level": self.level.value if self.level is not None else None,
            "detail": self.detail,
            "witness": self.witness.to_dict() if self.witness is not None else None,
            "bound": self.bound.to_dict() if self.bound is not None else None,
        }
