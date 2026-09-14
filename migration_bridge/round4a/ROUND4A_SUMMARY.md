# B Round 4A Summary

## Repository

- KTransformers Round 3 final / Round 4A base:
  `f0bfd6b9eb78b33871c9f2d2bb7fb9b06df5d9be`
- Round 3 tag: `round3-a3-verified`
- Round 4A branch: `feature/kt-multiexpert-multilayer-tp1`
- Round 4A implementation/evidence final:
  `03e72315b6e198a27357cb65520bf4f8ac09c134`
- SGLang repository: `kvcache-ai/sglang`
- SGLang Round 3 final / Round 4A base:
  `06b319dc3a62b77c880e36b042d273bfc3957d12`
- SGLang branch: `feature/kt-ep-multiexpert-tp1`
- SGLang Round 4A production changes: none
- Patch preservation: not applicable; child tree remains at its frozen base

## A3

- OS/kernel: openEuler aarch64
  `6.6.0-72.0.0.76.oe2403sp1.aarch64`
- CPU visibility: disposable container pinned to CPUs `0-15`
- NPU: Ascend 910, NPU 0 only; `npu-smi 26.0.rc1`
- Python: 3.11.15
- PyTorch: 2.9.0+cpu; torch_npu 2.9.0.post2
- Container: `kt-r3-dsv2lite`, 128 GiB limit
- Host/business containers: unchanged
- End state: inference server stopped; NPU 0 has no process

## Model

- Model: DeepSeek-V2-Lite
- Revision: `604d5664`
- Config SHA256:
  `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`
- Tokenizer SHA256:
  `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`

## Round 3 Replay

- Placement: Layer 17 / Expert 8
- CPU repeat: 10/10 byte-identical
- CPU-vs-BF16-rounded-FP32 relative L2: `5.29088030502319e-05`
- CPU-vs-NPU relative L2: `0.004299063928345015`
- Full-model A/B: 15/15 exact token IDs
- Prefix determinism: PASS
- Maximum `|delta logprob|`: `0.08105409145355225`
- Status: `A3_VERIFIED_READY`

## Prompt Corpora

- Selection S: 12 prompts; SHA256
  `6f743d4d8caa5f7480eee9dc03d4e3af4f647cdf255e1abd49ba18fc63bb10d2`
- Validation V: 9 prompts; SHA256
  `9ae547d3fef84f097b71eb944952e708298168be2083d1b1ce4faff76d03268e`
- Stability T: 6 prompts; SHA256
  `e82fe338c2785b0866dff7cd5a85f236620c72558ec758637fe8bbbcc595e700`

All prompt text and tokenizer input IDs are frozen. S, V and T are disjoint;
only S influenced placement.

## Route Profile

- MoE layers: 26 of 27 model layers
- Experts per MoE layer: 64
- Profile requests: 12, each greedy 32-token generation
- Frequency buffer shape: `[1000, 27, 64]`
- Total recorded routes: 80,496
- Raw frequency artifact SHA256:
  `cd405edb882b85c8a9651b428a7ec35ba5c3f04cd2bfdb1b6af67948fa1c9a22`

## Placement P1

- Layer 17 CPU experts: `{6,8,25,36}`
- Total CPU experts: 4
- Physical NPU experts at Layer 17: 60
- Placement SHA256:
  `9548bf6e06014e034c6a6650af3a891a546f70d8c8d79b059174ba571c44471f`
- GGUF SHA256:
  `a16a50827ec81b54195bf246c7f9d05f7c1d5f3601ee33426c732f65892e180f`
- Validation CPU hits: 1,233; all 4/4 experts hit
- Controlled multi-expert status: PASS
- Full-model status: BLOCKED

## Placement P2

- Layers: `{1,9,17,26}`
- CPU experts per layer: 4; total 16
- Placement SHA256:
  `f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3`
- Static validation: PASS
- A3 execution: `NOT_RUN_P1_BLOCKED`

## Placement P3

- Layers: `{1,5,8,12,17,19,22,26}`
- CPU experts per layer: 4; total 32
- Placement SHA256:
  `77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d`
- Static validation: PASS
- A3 execution: `NOT_RUN_P1_BLOCKED`

## Expert Identity Matrix

- P1 experts checked: 4/4
- Repeats: E6/E25/E36 five; E8 ten
- CPU determinism: all byte-identical
- CPU-vs-BF16-rounded-FP32 relative L2: min `0.0`, max
  `5.29088030502319e-05`
- CPU-vs-NPU relative L2: min `0.0038499988353798906`, median
  `0.004294382664898351`, max `0.004597487464640683`
- Status: PASS

## Same-Layer Multi-Expert

- CPU-not-hit: exact
- One CPU hit: explicit-reference exact
- Two CPU hits: explicit-reference exact
- Multi-token / different TopK positions: exact
- Routing weights: exactly once
- Routed scaling: exactly once
- Shared expert: exactly once
- Sequential versus overlap routed/final hashes: exact
- One wrapper, 1,000 forwards: one output hash
- Status: controlled layer `A3_VERIFIED_READY`

## Full-Model P1 A/B

- Validation requests: 45/45 succeeded
- Token exact: 30/45
- Prefix deterministic: PASS in both modes
- All logprobs finite: YES
- Maximum `|delta logprob|`: `2.4378519617021084`
- Status: BLOCKED

The earliest investigated divergence uses an identical history. All-NPU favors
token 185 over 549 by `0.0625`; P1 favors token 549 over 185 by `0.0625`.
The deterministic candidate-gap shift is `0.125`. Evidence supports numerical
near-tie amplification from the LLAMAFILE versus BF16 NPU expert paths under
high CPU-hit coverage, rather than routing/mapping/lifetime nondeterminism.

## Stability and Memory

- P1 real-layer 1,000-forward lifecycle: PASS
- P1 45-request process: no NaN/Inf, crash, deadlock or traceback
- Mandatory P3 campaigns and five-checkpoint leak study:
  `NOT_RUN_P1_BLOCKED`

## Performance Telemetry

- Corpus V all-NPU 45-request wall sum: 53.8209 s
- Corpus V P1 45-request wall sum: 109.9675 s
- No performance claim: YES

## Regression

- Placement/unit tests: 5 passed
- Round 3 P0 compact/full replay: PASS
- P1 controlled real routing and lifecycle: PASS
- Final Round2A/2B/2C/3/SGLang suite: `NOT_RUN_P1_BLOCKED`

## Frozen Scope

- TP: 1
- Graph: OFF
- Deferred experts: OFF
- Dynamic placement: OFF
- MTP: OFF
- Speculative decoding: OFF
- W8A8/MXFP4: OFF
- NUMA/threadpool scaling: OFF; threadpool count remains 1
- Old integrated Ascend operators restored: NO

## Sub-Gates

```text
ROUND3_REPLAY = A3_VERIFIED_READY
MULTI_EXPERT_SINGLE_LAYER = BLOCKED
MULTI_LAYER_P2 = NOT_RUN_P1_BLOCKED
MULTI_LAYER_P3 = NOT_RUN_P1_BLOCKED
```

## Final Gate

```text
DEEPSEEK_V2_LITE_TP1_MULTI_PLACEMENT = BLOCKED
```

## Remaining Blocker

```text
profile: P1
classification: TOKEN_DIVERGENCE / LOGITS
reproduction: Corpus V 45-request all-NPU versus P1 matrix
first investigated failing prompt/token: v_en_01 / generated step 1
expected: all-NPU token 185
actual: P1 token 549
candidate root cause: deterministic numerical near-tie flip caused by
  LLAMAFILE-vs-BF16-NPU expert-path differences under high-frequency CPU hits
next minimal experiment: quantify expert/layer output error on captured v_en_01
  hidden states and evaluate a higher-NPU-parity CPU numerical path; do not
  change placement, reduce CPU hits or relax acceptance thresholds
```

## Production Changes

No KTransformers or SGLang production source was changed. Round 4A added only
deterministic corpora/placement/export/verification tools, frozen artifacts,
evidence manifests, tests and audit documentation.
