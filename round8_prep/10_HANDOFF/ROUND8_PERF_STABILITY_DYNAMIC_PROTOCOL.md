# Round8 Q32 performance, stability and Dynamic protocol freeze

The Q64-C64 engineering reference and Q32-C64 target are already system and
CMMLU screened. Their model, tokenizer, F32 GGUF, NPU2, C64 two-subpool CPU
configuration and resident masks remain frozen. Diagnostic forward hooks are
removed for timed runs; no model arithmetic changes are made.

## Isolated static performance

Two fixed batch-one workloads use 64/256 input token IDs and
32/64 output tokens. The previous long workload is removed by the user's
2026-09-19 scope reduction. Qwen chat rendering has thinking disabled. Greedy
generation uses `ignore_eos` to obtain the exact output budget. Each profile
gets a warmup 8-token request before every measured block. The three-block
profile order is Q64, Q32, Q32, Q64 for each block (ABBA). Record client E2E
seconds, server E2E seconds, output count and hash. Model load time is excluded
from the measured request, but recorded separately. Every block requires fresh
host admission and no competing workload on the assigned NPU/CPU set. Compare
paired log ratios per workload with descriptive uncertainty; no speedup is
predeclared as a hard correctness gate. Times collected with forward hooks are
not formal performance evidence.

This reduced protocol therefore contains 12 clean-process ABBA cells and 24
measured requests. Campaigns v1-v4 remain immutable historical evidence under
the earlier, longer protocol and are not merged with the reduced v5 results.

## Long-run stability

After static performance, use the selected Q32 static profile without forward
hooks: six frozen prompts, 64 generated tokens, three cycles, two independent
campaigns (2304 tokens). Require identical token IDs across same prompt runs,
finite responses, no crash/deadlock and no monotonic unbounded RSS/HBM increase.
Record RSS/HBM before, peak and after each campaign. An allocator plateau is
distinguished from a leak. Run the same frozen configuration after clean restart.

## Qwen Dynamic qualification

Dynamic is a separate candidate on frozen Q32-C64, Deferred OFF. First audit
the actual Qwen model and KT wrapper trigger/update path. Freeze a Qwen-specific
SCHED_DEV and disjoint SCHED_HOLDOUT before collecting activation statistics.
Choose threshold on DEV only. Before held-out, freeze the threshold and initial
mask in a local commit. Check no-trigger and trigger paths, exact copied weight
identity, generation consistency across weights/mapping/masks, workload shift,
same-path determinism and clean restart. Evaluate the same frozen CMMLU 256
items using the same evaluator; observed score must be within 5pp of the Q64
reference with zero invalid outputs. Compare Dynamic with the best qualified
static baseline using isolated alternating performance blocks. A system/quality
pass without repeatable performance benefit is recorded as implemented but
default OFF.
