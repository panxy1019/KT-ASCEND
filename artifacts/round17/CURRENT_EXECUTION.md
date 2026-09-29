# Round17 current execution

Date: 2026-09-29

Status: `ROUND17_COMPLETE_NO_NEW_PROMOTION`

## Completed

- P0: frozen S0 identity restored; physical12 and NUMA2/3/4/7 admission used 2×10 seconds per start.
- P1: two valid diagnostic blocks, v512 and v514, each with six requests and 125 steady W2 Decode batches.
- A: complete QSA parent measured; no bounded mature candidate can reach the 20% complete-parent gate.
- B: larger exact segment screened; no safe bounded segment can reach the 15% complete-layer gate without a new state-aware graph runtime.
- Fresh S0: v519 and v520, profiler OFF, 10 generations each, W1/W2/W4 complete outputs.
- Final retirement: owned containers 0, physical12 processes 0, targeted registered-file cache retirement PASS.

## Final decision

- Performance promotion: `NONE`.
- Default: keep frozen S0.
- Fresh W2 TPOT: 259.926 / 265.632 ms.
- Fresh W2 serial output throughput: 3.5946 / 3.5206 tok/s.
- Candidate request ABBA: not run because no candidate passed the component-selection gate.

## Next action

None in Round17. Any future work must be a separately authorized complete fused-QSA backend or state-aware exact decoder runtime project.
