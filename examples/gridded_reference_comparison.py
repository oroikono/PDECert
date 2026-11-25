"""Compare a small stored grid with analytical values at the same nodes."""

import json
from itertools import product
from pathlib import Path

from pdecert import compare_reference_fields, load_grid


def main() -> dict:
    grid = load_grid(Path(__file__).with_name("rectilinear-grid.json"))
    points = list(product(*(values for _, values in grid.axes)))
    coordinates = {
        name: [point[index] for point in points] for index, name in enumerate(grid.coordinate_names)
    }
    return compare_reference_fields(
        coordinates,
        dict(grid.fields),
        {"u": [x + 2 * t for x, t in points]},
        problem_id=grid.problem_id,
        candidate_id=grid.artifact_id,
        reference_id="analytical-affine-nodes-01",
        reference_description="u=x+2*t evaluated at the stored dimensionless nodes",
        reference_uncertainty="Exact binary values for this manufactured example only",
        sampling_description="Cartesian product of stored axes; final axis varies fastest",
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
