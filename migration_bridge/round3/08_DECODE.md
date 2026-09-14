# Autoregressive Decode Verification

| Decode budget | English | Chinese | Structured numeric |
|---:|---:|---:|---:|
| 1 | PASS | PASS | PASS |
| 8 | PASS | PASS | PASS |
| 16 | PASS | PASS | PASS |
| 32 | PASS | PASS | PASS |
| 64 | PASS | PASS | PASS |

PASS means exact greedy token-ID agreement with the all-NPU baseline, finite
logprobs, and prefix consistency. KV-cache operation remained in SGLang and was
not duplicated by the KT bridge.

## Buffer lifetime result

The initial shared torch_npu D2H staging path had a nondeterministic decode race:
CPUInfer could see route IDs and weights before the private copy stream completed.
The final ACL-stream synchronization and synchronous ACL transfer path passed:

- 10/10 independent cold processes at `qlen=1`;
- 20 repeated decode calls in the formal regression;
- the full 44-test Round 2C matrix;
- the final 576-token full-model stability replay.

No shared host buffer is reused while a raw CPUInfer pointer may still access it.

