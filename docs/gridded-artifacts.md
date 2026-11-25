# Rectilinear nodal artifacts

`GridCandidate` stores already computed scalar fields on a Cartesian product of
coordinate axes. It gives numerical-solver exports and saved neural-operator
predictions a small, dependency-free representation before a grid-specific PDE
checker is available.

## Load and inspect a grid

```python
from pdecert import load_grid, write_grid

grid = load_grid("examples/rectilinear-grid.json")
print(grid.coordinate_names)  # ("x", "t")
print(grid.shape)             # (3, 2)
print(grid.field_names)       # ("u",)
write_grid(grid, "grid-copy.json")  # requires a new output file
```

The version-1 [JSON schema](../schema/grid-artifact-v1.schema.json) describes the
portable shape. The Python loader also checks relationships that JSON Schema
does not express: unique names, increasing axes, field lengths, finite binary64
values, and the total value budget. A schema pass alone is insufficient.

The Python constructor accepts axes and fields as ordered name/value pairs:

```python
from pdecert import GridCandidate

grid = GridCandidate(
    artifact_id="solver-export-01",
    problem_id="my-problem-01",
    axes=(("x", (0.0, 0.25, 1.0)), ("t", (0.0, 1.0))),
    fields=(("u", (0.0, 2.0, 0.25, 2.375, 1.0, 3.0)),),
    discretization="Dimensionless nodal samples with one deliberate value defect",
    provenance={"method": "manufactured example"},
)
```

Coordinates may be nonuniform. Each axis contains at least one finite value;
two or more values must be strictly increasing. Names are unique ASCII
identifiers within the axis and field namespaces. A field contains exactly the
product of the axis lengths. Coupled scalar fields share those same nodes.
Input lists are copied to immutable tuples, and provenance is copied before
storage. `grid_to_dict()` returns fresh containers for serialization.

## Storage and numerical semantics

Version 1 declares `value_location: "nodes"`,
`storage_order: "last_axis_fastest"`, and `dtype: "float64"`. For axis lengths
`(n0, ..., nd)` and indices `(i0, ..., id)`, the flat field index is
`sum(ij * product(nk for k > j))`. Equivalently:

```python
from itertools import product

points = list(product(*(values for _, values in grid.axes)))
```

For the example, nodes are `(0,0), (0,1), (0.25,0), (0.25,1), (1,0), (1,1)`.
Changing axis or value order changes the represented field. The reader never
sorts, transposes, interpolates, or fills missing values.

Only Python `int` and `float` values are admitted. Booleans, complex numbers,
nested values, framework arrays, and integers that lose precision on conversion
to binary64 are rejected. Convert external arrays explicitly after checking
their axes, units, and flattening order. Finite floats retain their represented
values; signed zero is normalized to positive zero. JSON decimal tokens use
binary64 rounding; nonzero tokens that underflow to zero and values that
overflow are rejected. The reader cannot recover precision already lost by a
caller before construction or by an external JSON parser before `grid_from_dict()`.

The file reader accepts at most 4,000,000 bytes of UTF-8 JSON. It rejects
duplicate object keys, nonstandard numeric constants, missing or extra fields,
and unsupported versions, locations, storage orders, or dtypes with
`GridArtifactError`. Across all axes and fields, at most 200,000 stored numbers
are accepted. The writer uses the same byte limit and refuses existing paths.
Ordinary filesystem errors remain filesystem errors. These limits bound the
accepted payload, not a hostile process's total runtime or memory.

## Compare values at the stored nodes

Run the complete example from a checkout:

```bash
python -m examples.gridded_reference_comparison
```

It explicitly expands the stored Cartesian nodes into coordinate columns for
the existing `compare_reference_fields()` API. The committed fixture is a
manufactured field `u=x+2*t` with an injected `+0.125` discrepancy at
`(x,t)=(0.25,1)`. The report records six samples, maximum absolute difference
`0.125`, that location at row 3, and `EMPIRICAL` evidence. It issues no PDE
verdict. The comparison sorts column names for reporting while preserving rows.

That API has its own limit of 1,000,000 expanded numeric values. A grid that
fits the storage budget may exceed the comparison budget; the comparison then
fails explicitly. The example needs only core dependencies and performs no
model inference, training, finite differencing, or quadrature.

## Scope and next extensions

Discretization descriptions, problem IDs, and provenance are caller declarations.
They do not establish compatible equations, coordinate units, trusted solver
execution, generator identity, or independent reference truth. Preserve solver
versions, precision, coordinate/field units, source digests, and generation
settings in the description or provenance when exporting a real calculation.
A run manifest can bind the exact grid file and report bytes separately.

Version 1 excludes cell-centered or staggered values, curvilinear and unstructured
meshes, sparse/missing nodes, complex fields, periodic endpoint conventions,
and automatic array or checkpoint loading. Values may be discontinuous, but
the artifact assigns no classical, weak, or entropy solution semantics.
`verify_artifact()` rejects grids as an unsupported problem/artifact pair.
Atlas v2 currently accepts symbolic and frozen-callable records only.

A grid PDE checker needs a separate contract for reconstruction, derivative
stencils, boundary treatment, refinement, and discretization error. A sampled
reference discrepancy is neither a PDE counterexample nor a continuous-domain
solution-error bound. See [ADR-0011](adr/0011-rectilinear-nodal-artifacts.md).
