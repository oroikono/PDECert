"""Portable nodal values on a finite rectilinear grid."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar


GRID_ARTIFACT_VERSION = 1
GRID_ARTIFACT_MAX_VALUES = 200_000
GRID_ARTIFACT_MAX_BYTES = 4_000_000
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FORMAT = {
    "schema_version": GRID_ARTIFACT_VERSION,
    "artifact_kind": "rectilinear_grid",
    "value_location": "nodes",
    "storage_order": "last_axis_fastest",
    "dtype": "float64",
}
_KEYS = set(_FORMAT) | {
    "artifact_id",
    "problem_id",
    "axes",
    "fields",
    "discretization",
    "provenance",
}


class GridArtifactError(ValueError):
    """A grid artifact is malformed or outside the supported representation."""


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GridArtifactError(f"{path}: expected a nonblank string")
    return value


def _object(value: object, keys: set[str], path: str) -> Mapping:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise GridArtifactError(f"{path}: expected an object with string keys")
    if set(value) != keys:
        raise GridArtifactError(f"{path}: expected exactly these fields: {', '.join(sorted(keys))}")
    return value


def _columns(value: object, path: str) -> list[tuple[str, list | tuple]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise GridArtifactError(f"{path}: expected a nonempty sequence of named columns")
    columns = []
    names = set()
    for index, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise GridArtifactError(f"{path}[{index}]: expected a name and numeric sequence")
        name, values = pair
        if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None:
            raise GridArtifactError(f"{path}[{index}].name: expected an ASCII identifier")
        if name in names:
            raise GridArtifactError(f"{path}: names must be unique")
        names.add(name)
        if not isinstance(values, (list, tuple)) or not values:
            raise GridArtifactError(f"{path}.{name}: expected a nonempty flat numeric sequence")
        columns.append((name, values))
    return columns


def _numbers(values: list | tuple, path: str) -> tuple[float, ...]:
    result = []
    for index, value in enumerate(values):
        location = f"{path}[{index}]"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GridArtifactError(f"{location}: expected a finite real int or float")
        try:
            number = float(value)
        except OverflowError as error:
            raise GridArtifactError(f"{location}: value exceeds binary64") from error
        if not math.isfinite(number):
            raise GridArtifactError(f"{location}: expected a finite value")
        if isinstance(value, int) and int(number) != value:
            raise GridArtifactError(f"{location}: integer loses precision in binary64")
        result.append(0.0 if number == 0 else number)
    return tuple(result)


@dataclass(frozen=True)
class GridCandidate:
    """Immutable nodal fields; the final coordinate axis varies fastest.

    Every field has ``prod(shape)`` values at the Cartesian product of the
    declared axes. Loading values does not reconstruct derivatives, bind a
    problem template, or establish a PDE solution semantics.
    """

    artifact_id: str
    problem_id: str
    axes: tuple[tuple[str, tuple[float, ...]], ...]
    fields: tuple[tuple[str, tuple[float, ...]], ...]
    discretization: str
    provenance: Mapping[str, str]
    kind: ClassVar[str] = "rectilinear_grid"

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.problem_id, "problem_id")
        _text(self.discretization, "discretization")
        axes = _columns(self.axes, "axes")
        fields = _columns(self.fields, "fields")
        count = math.prod(len(values) for _, values in axes)
        stored = sum(len(values) for _, values in axes + fields)
        if stored > GRID_ARTIFACT_MAX_VALUES or count * len(fields) > GRID_ARTIFACT_MAX_VALUES:
            raise GridArtifactError("grid exceeds the maximum numeric value budget")
        if any(len(values) != count for _, values in fields):
            raise GridArtifactError(f"fields: expected {count} values per field for the grid shape")
        normalized_axes = []
        for name, values in axes:
            numbers = _numbers(values, f"axes.{name}")
            if any(left >= right for left, right in zip(numbers, numbers[1:])):
                raise GridArtifactError(f"axes.{name}: coordinates must be strictly increasing")
            normalized_axes.append((name, numbers))
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise GridArtifactError("provenance: expected a nonempty string mapping")
        provenance = dict(self.provenance)
        for key, value in provenance.items():
            _text(key, "provenance key")
            _text(value, f"provenance.{key}")
        object.__setattr__(self, "axes", tuple(normalized_axes))
        object.__setattr__(
            self,
            "fields",
            tuple((name, _numbers(values, f"fields.{name}")) for name, values in fields),
        )
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return axis lengths in the declared coordinate order."""

        return tuple(len(values) for _, values in self.axes)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Return coordinate names in storage order."""

        return tuple(name for name, _ in self.axes)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return field names in their declared order."""

        return tuple(name for name, _ in self.fields)


def grid_from_dict(value: object) -> GridCandidate:
    """Validate a version-1 grid payload without evaluating its fields."""

    payload = _object(value, _KEYS, "grid")
    for key, expected in _FORMAT.items():
        if type(payload[key]) is not type(expected) or payload[key] != expected:
            raise GridArtifactError(f"{key}: expected {expected!r}")

    def columns(key: str, values_key: str) -> tuple:
        entries = payload[key]
        if not isinstance(entries, list):
            raise GridArtifactError(f"{key}: expected an array")
        result = []
        for index, entry in enumerate(entries):
            item = _object(entry, {"name", values_key}, f"{key}[{index}]")
            if not isinstance(item[values_key], list):
                raise GridArtifactError(f"{key}[{index}].{values_key}: expected an array")
            result.append((item["name"], item[values_key]))
        return tuple(result)

    return GridCandidate(
        artifact_id=payload["artifact_id"],
        problem_id=payload["problem_id"],
        axes=columns("axes", "coordinates"),
        fields=columns("fields", "values"),
        discretization=payload["discretization"],
        provenance=payload["provenance"],
    )


def grid_to_dict(artifact: GridCandidate) -> dict[str, object]:
    """Return fresh JSON containers while preserving axis and field order."""

    if not isinstance(artifact, GridCandidate):
        raise TypeError("artifact must be a GridCandidate")
    return {
        **_FORMAT,
        "artifact_id": artifact.artifact_id,
        "problem_id": artifact.problem_id,
        "axes": [{"name": name, "coordinates": list(values)} for name, values in artifact.axes],
        "fields": [{"name": name, "values": list(values)} for name, values in artifact.fields],
        "discretization": artifact.discretization,
        "provenance": dict(artifact.provenance),
    }


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise GridArtifactError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _json_float(token: str) -> float:
    number = float(token)
    if not math.isfinite(number):
        raise GridArtifactError("JSON number exceeds binary64")
    mantissa = token.lower().partition("e")[0]
    if number == 0 and any(digit in "123456789" for digit in mantissa):
        raise GridArtifactError("nonzero JSON number underflows binary64")
    return number


def _reject_constant(token: str) -> None:
    raise GridArtifactError(f"nonfinite JSON constant: {token}")


def load_grid(path: str | Path) -> GridCandidate:
    """Load bounded, strict UTF-8 JSON; no array or model backend is imported."""

    with Path(path).open("rb") as source:
        data = source.read(GRID_ARTIFACT_MAX_BYTES + 1)
    if len(data) > GRID_ARTIFACT_MAX_BYTES:
        raise GridArtifactError(f"grid file exceeds {GRID_ARTIFACT_MAX_BYTES} bytes")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_float=_json_float,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise GridArtifactError(f"invalid grid JSON: {error}") from error
    return grid_from_dict(payload)


def write_grid(artifact: GridCandidate, path: str | Path) -> None:
    """Write deterministic UTF-8 JSON to a new file; existing files are preserved."""

    data = (
        json.dumps(grid_to_dict(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(data) > GRID_ARTIFACT_MAX_BYTES:
        raise GridArtifactError(f"grid JSON exceeds {GRID_ARTIFACT_MAX_BYTES} bytes")
    with Path(path).open("xb") as destination:
        destination.write(data)
