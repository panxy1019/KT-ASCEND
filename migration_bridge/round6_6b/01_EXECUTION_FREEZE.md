# Round6.6B execution freeze

Input SHA256:
`885ccf76386dcfed2867142417300ac69c4401754360f92e6b6e4f179adffd97`.
Full machine-readable input/exemplar/source manifest is in`corpora/`.
Longest input971tokens;2048context and512GSM generation cap unchanged.
No normalized question-text overlap with V1. One generic Chinese question stem
occurs twice with different subjects/options;these are distinct questions.

Executable SHA256:

| Script | SHA256 |
|---|---|
|run_base_protocol.py|2fe6b34e3cc7296ec73f324636d5477ad714598153146e47188751c070052f9b|
|run_sequence.py|2c82a4cfc7fee947382b74bcfbcbaeba766a4899713dfc23dae44177fa141518|
|audit_screen.py|2ba7a1d983e0f924e43e9fd4b96e25857bf301da6b2fe44ed2cec74383393e9d|

Sequence uses a3-hour owned-process guard,port31000,NPU0 only in the dedicated
`kt-r65-static-offload` container. It runs All-NPU then four-R32 serially. Only
newly owned process groups can be terminated. Unrelated container`verl` and host
libraries are untouched. Runtime provenance remains the91-file Round6.6 B0
manifest plus the previously frozen model/tokenizer/profile/parser assets.

Remote outputs:
`/workspace/results/r65-r66b-base-protocol-v1/`.
Outer guard:
`/workspace/results/r65-r66b-base-protocol-guard-v1/`.

Validation before collection:4scoring unit tests;after input freeze:5/5 including
hash,exclusion,counts,example count and context checks. PriorRound6.6 tests16/16
andRound6.6A tests2/2 also pass. A later local fixture-test addition is not a
production/runtime change and is not used by the running collector.

Final outcome must be read from completed paired results and raw-response audit,
not inferred from the fact that the job launched. Historical V1 remains failed.
