# F1 Static Frequency — bounded full-model gate

Date: 2026-09-21. Status: FULL_MODEL_SMOKE for the exact combination below.
This completes the F1 framework gate, not product quality or performance.

## Qualified scope

- Qwen3.8 original verified checkpoint,48MoE layers,512logical experts/layer.
- C128,4NUMA pools, MXFP4 complete CPU backing, original BF16 NPU promotion source.
- Physical Q64 resident capacity per layer; Static Frequency placement.
- Serial eager hybrid execution, synchronous PLE, BF16 context cache.
- Framework overlayv3, route collection enabled for observable map evidence.
- Host110.123.0.3, exclusivephysical15, fresh admission and live resource guard.

No Q32 physical, Frequency+Dynamic, other CPU bank, Layerwise, AsyncPLE, long-context,
CMMLU256, performance or deployment qualification is conferred by this result.

## Actual evidence

| Trial | Purpose | Result |
|---|---|---|
| 24 | New overlay, all new flagsOFF; DEV2+HOLDOUT2 | PASS; known-answer391, finite responses |
| 25 | Real logical-route collection forDEV2 | PASS;47input tokens,48complete layers; identical output IDs and zero logprob difference vs24 |
| 26 | Generated FrequencyQ64 load, HOLDOUT2, activeabort/reuse | PASS; all48actual maps equalmanifest; canary391; state reuse retains baseline output |
| 27 | Fresh process replay of same Frequencymanifest, HOLDOUT2 | PASS; all48maps and output token trajectories equal26 |

Observed owned-process HBM peaks: trial24=39848MiB,25=39842MiB,
26=39846MiB,27=39844MiB. These are bounded-run resource observations, not a
performance comparison. No resource guard stopped these trials.

Evidence lives under `evidence/physical15/20260920-full-model-v24` through `v27`.
Each `framework-summary.json` binds the relevant raw files by SHA256. Namespace
dates retain the existing trial convention; actual event times are in receipts.
Frequency outputs are in `evidence/physical15/20260921-framework-frequency-v1`.

## Determinism and identity

The47input tokens are26+21 for the two hand-authored DEV prompts. The generation
tool checks their request IDs, input-ID hashes, complete layer coverage and count
conservation against the HTTP evidence. No HOLDOUT output or CMMLU material is
used to construct placement.

Counts are ranked independently per layer. Count ties resolve to ascending logical
expert ID. Selected IDs are then sorted to establish repeatable physical slots.
Both Q32 andQ64 manifest reconstruction matched the saved documents exactly.
Q32 here is only a generated mask, not a device-qualified allocation.

Actual Q64Frequency changes2578 resident experts across48layers compared with the
observed Uniform producer maps. Thus the real path is not a uniform fallback.
Each manifest binds model manifest, tokenizer files, runtime payload, corpus and
producer placement. Loader SHA validation rejects changed manifest bytes.

## Failure handled during this gate

The first trial25 resource admission rejected low selected-node free memory after
trial24 ended. Only the48 owned MXFP4 bank file caches were advised DONTNEED, with
file size/mtime/inode checks. The same admission passed afterward. No global cache
drop, resource threshold change, unrelated process termination or timer restart
was used. Details are in44_FRAMEWORK_F1_IMPLEMENTATION_NOTES.md and the remote
`20260921-framework-owned-mxfp4-reclaim-v*.json` records.

## Remaining limits

The tiny DEV corpus establishes an executable deterministic pipeline, not a
representative hot-set policy. Restart and same-placement checks demonstrate the
tested trajectories only. Frequency changes which low-bit CPU computations are
replaced by original-BF16 NPU computations, so cross-placement numerical identity
and population-level quality are not assumed. Those comparisons remain in the
qualification backlog. The final interaction matrix must rerun the applicable
short gate on the final combined framework binary.
