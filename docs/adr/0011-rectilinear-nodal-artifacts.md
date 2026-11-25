# ADR-0011: Rectilinear nodal artifact representation

**Status:** Proposed, with an experimental implementation for review
**Date:** 2026-11-25

## Context

Saved numerical fields need explicit coordinates and storage order before a
checker can interpret them. Turning a grid into an autodiff callable would add
unstated interpolation and derivative assumptions. The existing artifact
protocol can describe a grid without claiming a new verification backend.

## Decision

Add `GridCandidate` in `gridded.py` and a separate version-1 JSON format. Store
ordered, nonempty rectilinear axes and named flat scalar fields at their nodal
Cartesian product. The final coordinate axis varies fastest. Coordinate axes
are strictly increasing, singleton axes are allowed, and field lengths equal
the product of the axis lengths. The format declares node location, storage
order, binary64 values, discretization description, IDs, and caller provenance.

Validation freezes input containers, rejects malformed or unsupported payloads,
and bounds file bytes and total stored numeric values. Array conversion stays
explicit. Loading imports no numerical framework and executes no candidate code.
The writer creates a new file and preserves existing destinations.

Reuse `SolutionArtifact` through its existing `kind` and `field_names` contract.
Do not register a grid checker or admit a grid to Atlas v2 in this change.
An example expands the nodes explicitly for the existing empirical reference
comparison API. No new core or optional dependency is required.

## Alternatives

- Opaque arrays with shape metadata omit coordinate and ordering checks.
- A universal mesh abstraction would require topology, location, and geometry
  contracts beyond this first numerical representation.
- Automatic interpolation into the callable lane would change the mathematics
  and silently assign derivatives to discrete data.

The restricted nodal format supplies an independently usable boundary for later
solver adapters and numerical checkers while keeping those assumptions open.

## Evidence and compatibility

Validation establishes representation consistency only. Neither declared
provenance nor a problem ID proves execution, template compatibility, solution
regularity, or a PDE obligation. Stored discontinuities do not define a weak or
entropy solution. Reference comparisons remain empirical values at supplied
nodes; derivative, quadrature, convergence, and solution-error claims require
separate numerical analysis and implementation.

Existing case, template, report, and Atlas schemas retain their current meanings.
Unsupported grids continue to fail explicitly at `verify_artifact()`. The
gridded release gate remains incomplete until its evaluation and benchmark
requirements have reproducible evidence.
