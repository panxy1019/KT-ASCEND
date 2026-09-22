# Recorder pilot recovery

Attempt1 runtime: `/workspace/c64-dev/round7/runtime-v1/`.
Corpus was frozen and committed before this pilot. No DEV activation fit,
Frequency mask, HOLDOUT or quality evaluation was performed.

## Observed — A3_VERIFIED

The isolated 71-input-token, 8-output-token pilot produced rank3 stat data
`[9,27,64]`. Layer0 total0; every routed layer total474. Per-forward layer1
counts were `[426,6,6,6,6,6,6,6,6]`. All requested output IDs were finite and
the two sanity responses were identical. The collector's budget-1 assumption
expected468 and correctly rejected the dump. Owned guard reaping was complete.

This establishes one extra single-token pass, not its token/history identity.
An overlap scheduler's queued unused final decode pass is a mechanism hypothesis,
not yet a proven classification. The production path has not been changed.

## Test-only recovery

New attempt2 preserves attempt1 rather than overwriting it. Metadata is captured
at the official real recorder's `_on_forward_pass_start`: input IDs, positions,
forward ID, recorder epoch. Stat and metadata must be one-to-one. Each prefill
must equal a frozen prompt; each decode input must equal a returned token at
its exact position. Missing/duplicate/unknown passes are rejected.

If the sole extra pass consumes the final already-returned token, its raw stat
row remains preserved but is not fitted as an evaluated generation position.
Published activation counts retain prefill plus output-budget-1 consumed tokens.
Excluded row indices and the full raw stat are SHA-audited. This is an
orchestration/association correction, not a numerical acceptance relaxation.

Local regression: valid unused final row attributed/excluded; unknown and missing
rows rejected. Same-process observer ON/OFF trajectory parity is checked before
the repeat recorder pilot. No full DEV run starts until this attribution passes.
