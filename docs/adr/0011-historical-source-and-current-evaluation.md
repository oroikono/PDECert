# ADR-0011: Separate historical source identity from current evaluation

**Status:** Accepted for this implementation
**Date:** 2026-09-15
**Deciders:** Maintainer-directed local architecture review

## Context

The Fisher--KPP integrity record binds 18 training, input, and evaluator files.
The strict validator formerly resolved both the artifact and these historical
files in one checkout. Correctly changing the symbolic sampler therefore made
historical source validation fail. Updating frozen hashes would erase the
recorded source identity; ignoring them would weaken the replay contract.

## Decision

Add an explicit optional `historical_source_root` to the existing integrity
validator. The artifact remains bound to `repository_root`; every historical
source remains mandatory and must match its recorded digest in the explicit
source root. Omitting the new argument retains the original strict behavior.
No source lookup fetches files, imports archives, or chooses a revision silently.

Preserve a small data-only snapshot of all 18 exact files under
`benchmarks/historical/fisher-kpp-source-v1`. Its metadata identifies the full
revision supplying these matching bytes, without representing that revision as
the original training commit. The frozen artifact, both historical integrity
records, and historical evaluation result are unchanged.

A separate current Fisher--KPP runner emits a version-1 receipt. It first checks
all historical bytes and binds the active artifact, integrity record, template,
raw proposal, case and provenance record. Active mathematical inputs must still
occupy the historical bound paths and match their hashes. It then independently
hashes every Python source in the current package, explicit runner sources,
source-receipt schema and project metadata. It rejects a different imported
checkout and checks the same inputs and sources after the operation.

Inspection defaults to content validation without PyTorch. Explicit evaluation
uses the current trusted runner, writes fresh diagnostics and binds their digest,
configuration and runtime separately. Output files are created exclusively.
Every receipt states `historical_replay: false` and
`integrity_scope: content_identity_only`.

## Options considered

- Refresh historical source hashes: rejected because it rewrites the record.
- Stop checking evaluator files: rejected because it weakens the declared closure.
- Move changed code outside the bound modules: rejected because it evades closure.
- Require an implicit Git checkout/download on every validation: rejected because
  offline operation, source distributions and user-controlled provenance need
  explicit inputs.
- Explicit historical source root plus independent current receipt: selected;
  the supplied root can be the packaged snapshot or another matching archive.

## Consequences

Source evolution is possible without pretending it reproduces the historical
evaluator. Missing or changed historical files still fail. New or changed current
package modules invalidate old current-source receipts. Package-wide hashing is
intentionally conservative and may invalidate a receipt after an unrelated
module changes.

The source snapshot is not a complete historical runtime. A full historical
replay needs a complete matching checkout and the recorded dependency environment.
Library versions and a missing dependency lock are reported honestly. Source-file
hashes, module-origin checks and pre/post comparisons are not in-memory code
attestation, a concurrency-safe security sandbox, or proof of trusted execution.
External dependencies are identified by runtime versions, not their full byte
closure. Finite diagnostic success never becomes proof and no human labels change.

## Action items

1. [x] Preserve and verify the 18-file historical snapshot.
2. [x] Implement explicit source roots and versioned current receipts.
3. [x] Cover positive, tampered, missing and unbound inputs without PyTorch.
4. [x] Document separate identity, historical replay and current-evaluation commands.
5. [ ] Run optional numerical regeneration in an available recorded CPU Torch environment.
