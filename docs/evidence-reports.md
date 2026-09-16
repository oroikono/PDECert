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

### Refutation takes precedence

`report_from_dict` and `load_report` reject a `PROVED` or `INCONCLUSIVE`
summary if any event has outcome `REFUTED`, regardless of event order or
whether that refutation is exact or empirical. The public JSON schema enforces
this rule too. A `REFUTED` report may retain earlier discharged or abstained
obligations; the Python loader also checks that its decision witness and
evidence level match a refuting event.

An earlier abstention or sampled pass does not prevent a later exact discharge.
Those events and their incomplete reasons remain useful history and are not
refutations. No sampled pass is promoted to proof.

For example, a saved refutation cannot be hidden by changing only its summary:

```python
from pdecert import (
    ReportSchemaError, bind_symbolic_candidate, load_template, report_from_dict, verify,
)

template = load_template("examples/heat-template.json")
case = bind_symbolic_candidate(template, {"u": "0"})
payload = verify(case.problem, case.candidate_fields).to_dict()
assert payload["status"] == "REFUTED"  # The zero field violates the initial trace.

payload.update(status="INCONCLUSIVE", decision_evidence=None, witness=None)
try:
    report_from_dict(payload)
except ReportSchemaError as error:
    print(error)  # $.evidence_events: refuting evidence requires a REFUTED report
else:
    raise AssertionError("contradictory report was accepted")
```

Loading checks the report's internal consistency, not the mathematical truth of
its events. A standalone report does not contain the trusted problem's complete
obligation set. Neither loader nor schema validation proves that all required
obligations were recorded or that a claimed witness is numerically reliable;
retain the problem and replay the evaluator for those checks. Constructing a
`Report` or calling `to_dict`/`dump_report` alone is not report validation.

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
- refuting evidence hidden behind a `PROVED` or `INCONCLUSIVE` summary;
- unknown report or aggregation-policy versions; and
- non-standard `NaN` or `Infinity` JSON constants.

The complete contract is
[`schema/report-v1.schema.json`](../schema/report-v1.schema.json), and the design
rationale is [`ADR-0009`](adr/0009-versioned-evidence-reports.md).
