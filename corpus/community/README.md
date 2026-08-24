# Community atlas

This directory is the merge-friendly intake corpus for community PDE failure
records. It is separate from the immutable 20-record pilot.

Each accepted pending record occupies one directory:

~~~text
records/<record-id>/
├── record.json
├── case.json
└── raw-output.txt
~~~

The directory name must match the record ID. record.json contains provenance,
the raw-output digest, and annotation state. case.json uses the latest PDECert
problem schema. raw-output.txt is preserved byte-for-byte.

The initial synthetic records are mechanism-isolating fixtures, not natural
model outputs and not human ground truth. Their annotations stay pending until
the published review protocol is completed.

Validate the complete directory before opening a pull request:

~~~bash
pdecert corpus validate corpus/community
~~~

The validator rejects loose files inside records/, unexpected or symlinked
bundle entries, mismatched IDs, modified raw outputs, invalid embedded cases,
and completed labels that do not satisfy the published review protocol.
