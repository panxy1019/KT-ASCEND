# Round 3 Replay

Status: `A3_VERIFIED_READY`.

The replay used the frozen Layer 17 / Expert 8 placement and exact Round 3
model revision, F32 GGUF and runtime invariants.

- CPU repeats: 10/10 byte-identical; maximum repeat difference `0.0`.
- CPU vs BF16-rounded FP32 relative L2: `5.29088030502319e-05`.
- CPU vs NPU relative L2: `0.004299063928345015`.
- Full-model matrix: 15/15 requests completed and token IDs were exact.
- Prefix determinism: PASS for 1/8/16/32 against each 64-token trajectory.
- Maximum `|delta logprob|`: `0.08105409145355225`.
- All outputs/logprobs finite; no crash, traceback, deadlock or invalid route.

```text
ROUND3_REPLAY = A3_VERIFIED_READY
```
