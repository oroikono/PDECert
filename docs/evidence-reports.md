# Evidence reports

PDECert reports both a conservative decision and the evidence produced for each
represented obligation. The decision answers whether the available checks prove,
refute, or cannot decide the candidate. The evidence stream explains why.

## Read a report

```python
from pdecert import load_case, report_from_dict, verify

case = load_case("examples/exact_heat.json")
report = verify(case.problem, case.candidate_fields)
payload = report.to_dict()

assert payload["report_version"] == 1
assert payload["aggregation_policy_version"] == 1
assert report_from_dict(payload).to_dict() == payload

for event in payload["evidence_events"]:
    print(event["obligation_id"], event["kind"], event["outcome"])
```

The stable obligation identifiers come from the verification context:

- `constraint:0`, `constraint:1`, and so on refer to residual or condition
  obligations in declared order;
- `domain:<field>:<variable>` refers to one field/domain check.

`exact_checks` and `decision_evidence` remain compatibility summaries. New
integrations should retain and inspect `evidence_events`.

## Interpret evidence conservatively

| Kind | Outcome | Can produce `PROVED`? | Required payload |
| --- | --- | ---: | --- |
| `EXACT_CERTIFICATE` | `DISCHARGED` | Yes | Checker, obligation, explanation |
| `RIGOROUS_BOUND` | `DISCHARGED` | Yes | Quantity, bound type, norm, scope, assumptions, constants |
| `EMPIRICAL_COUNTEREXAMPLE` | `REFUTED` | No; it can produce `REFUTED` | Replayable witness |
| `EMPIRICAL_PASS` | `OBSERVED_PASS` | No | Sampling explanation |
| `ABSTENTION` | `ABSTAINED` | No | Reason |

A rigorous `UNIFORM_RESIDUAL` or `BOUNDARY_TRACE` bound is not a
`SOLUTION_ERROR` guarantee. Consumers should filter on `bound.bound_type`, not
only on the top-level `RIGOROUS_BOUND` label.

## Implement a rigorous checker

A checker may claim `RIGOROUS_BOUND` only when every declared proof has a
structured event. This example describes the shape; it is not itself a bound
computation:

```python
from pdecert import (
    BoundEvidence,
    BoundType,
    CheckResult,
    EvidenceEvent,
    EvidenceKind,
    EvidenceLevel,
    EvidenceOutcome,
)

obligation = "constraint:0"
bound = BoundEvidence(
    bound_type=BoundType.UNIFORM_RESIDUAL,
    quantity="absolute PDE residual",
    upper_bound=1e-8,
    norm="L_inf",
    scope="x in [0, 1]",
    assumptions=("outward-rounded interval evaluation",),
    constants={"precision_bits": 128},
)
result = CheckResult(
    proved_obligations=frozenset({obligation}),
    proof_level=EvidenceLevel.RIGOROUS_BOUND,
    evidence_events=(
        EvidenceEvent(
            obligation_id=obligation,
            checker="interval_residual",
            kind=EvidenceKind.RIGOROUS_BOUND,
            outcome=EvidenceOutcome.DISCHARGED,
            level=EvidenceLevel.RIGOROUS_BOUND,
            detail="validated enclosure covers the declared domain",
            bound=bound,
        ),
    ),
)
```

The checker must compute and justify the bound. Constructing this object does
not make an unvalidated number rigorous.

## Compatibility and unsupported cases

Existing exact checkers using `proved_obligations` and `proof_level=EXACT`
continue to run; the orchestrator creates an explicitly labeled compatibility
event. New checkers should emit events directly.

Version 1 deliberately rejects:

- a rigorous-bound claim without quantity and scope metadata;
- empirical evidence that claims to discharge an obligation;
- a refuting event without a witness;
- unknown report or aggregation-policy versions; and
- non-standard `NaN` or `Infinity` JSON constants.

The complete contract is
[`schema/report-v1.schema.json`](../schema/report-v1.schema.json), and the design
rationale is [`ADR-0009`](adr/0009-versioned-evidence-reports.md).
