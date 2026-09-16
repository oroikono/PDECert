"""A deliberately perturbed analytical field, not a trained-model audit."""

import json
import math
from itertools import product

from pdecert import compare_reference_fields


def run():
    points = list(product((-4.8, -2.4, 0.0, 2.4, 4.8), (0.1, 0.55, 1.0, 1.45, 1.9)))
    reference = [(1 + math.exp(x / math.sqrt(6) - 5 * t / 6)) ** -2 for x, t in points]
    candidate = [value + 0.02 * t for value, (_, t) in zip(reference, points)]
    return compare_reference_fields(
        coordinates={"x": [x for x, _ in points], "t": [t for _, t in points]},
        candidate={"u": candidate},
        reference={"u": reference},
        problem_id="fisher-kpp-classical-01",
        candidate_id="deliberate-0.02t-field-perturbation",
        reference_id="fisher-kpp-traveling-front-formula",
        reference_description="u=(1+exp(x/sqrt(6)-5*t/6))**(-2), evaluated with Python math",
        reference_uncertainty=(
            "Binary64 evaluation of an analytical formula; rounding is not rigorously bounded. "
            "This comparison does not verify the formula against the PDE."
        ),
        sampling_description=(
            "25 supplied interior points: x in [-4.8,-2.4,0,2.4,4.8], "
            "t in [0.1,0.55,1,1.45,1.9], x outer and t inner. Not a convergence study."
        ),
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
