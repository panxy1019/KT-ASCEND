# Round 6 D1 HD held-out decision

Evidence state: `HELDOUT_VERIFIED_REJECTION`.

The Round 6 Deferred numerical candidate frozen at parent commit
`469d69aa2c872cb7a091abccbb7164c7ba6adbe9` was evaluated once on the
independent HD corpus. No bound, candidate K, placement, expert set, prompt,
history, model, weight artifact, or execution mode was changed after seeing
HD output.

## Frozen inputs

| Item | Value |
| --- | --- |
| HD corpus SHA | `0ee3b50c9d02515f1492463ea1af5d507ad72267d43d5e0962e6bf9c6619adba` |
| History canonical SHA | `25b3160053db63527202119c41b7883faad577a6edcdaaee20eaf314c6617f19` |
| Candidate contract canonical SHA | `d5b228befe5bfc08554b7f9d1c87b4ff6d2348913d2cc94d73a9b3710ca27f26` |
| Pairwise bound | `0.46875` (`1.25 × QD max`) |
| Candidate K | `32` |
| Full-logit max-abs bound | `1.015625` |
| Full-logit relative-L2 bound | `0.05556806921958923` |

Both defer=0 and defer=1 captured all 16 prompts × 64 positions = 1024
positions on the same defer=0 history. Prompt order, token indices, corpus
SHA and history SHA were identical. Route telemetry was position-aligned;
defer=1 contained 471 target-row CPU-hit positions and 102 actual Deferred
target-row positions. All requests drained the pending set and all logits
were finite.

## Held-out result

| Gate | Result |
| --- | --- |
| Stable top-1 | `888/888 exact` |
| Ambiguous membership | `136/136` |
| Full-logit max-abs overflow | `0` |
| Full-logit relative-L2 overflow | `0` |
| Pairwise overflow | **`1`** |
| Final candidate status | **`HD_REJECTED`** |

The single hard failure is:

```text
prompt       = hd_en_02
token index  = 43
anchor token = 21
candidate    = 16520
baseline margin = 10.75
defer=1 margin  = 11.5
absolute pairwise distortion = 0.75
frozen bound = 0.46875
```

The serving top-1 remained token 21, so this is not a token flip. It is still
a hard rejection because the predeclared contract bounds pairwise distortion
over the union top-32 candidate universe; changing that rule after observing
HD would be held-out fitting. At this request the target token row had no CPU
hit, but the prompt-prefix portion contained 34 CPU routes in both modes. The
defer=1 prefix included one actual Deferred route at Layer 7. Therefore this
is not a valid whole-request CPU-not-hit control.

HD also contained three top-1 changes (`hd_en_02:37`, `hd_json_01:2`, and
`hd_json_01:3`). All three occurred inside the frozen ambiguity set, so none
was the rejecting gate. Raw top-1 agreement was 1021/1024.

## Distribution

| Metric | p50 | p95 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pairwise absolute distortion | 0 | 0 | 0.125 | 0.25 | **0.75** |
| Full-logit max absolute error | 0 | 0.25 | 0.542695 | 0.678789 | 0.8125 |
| Full-logit relative L2 | 0 | 0.00833062 | 0.0191027 | 0.0259215 | 0.0319061 |

The compact first-overflow artifact is
`evidence/hd/first_overflow.json`; its canonical payload SHA is
`c495c6c7e3c53173fff15462767d062d5b01bab7dcd2eafc165d4f815d8b935f`.
The complete validation is `evidence/hd/hd_validation.json`.

Raw trace SHA256 values retained on A3:

```text
defer=0 be7acd3116891a4ba66e7b1566739955965842619a621713d0eedf8d1c52f04b
defer=1 84c3a184dd49dbaf219506a96a1665abf6179ecb3c8ed99ea68715d68a229f78
```

## Decision and stop boundary

```text
D1_DEFERRED_PAIRWISE_CONTRACT = HELDOUT_REJECTED
D1 = BLOCKED_AT_HD_PAIRWISE_ENVELOPE
D2 = NOT_RUN
D3 = NOT_RUN
ROUND6_DEFERRED_EXPERT_ASCEND = BLOCKED
```

The implementation and D1 determinism evidence remain valid; a failed
numerical qualification does not retroactively turn them into failures. But
the approved order requires stopping before FD, downstream quality, D2 and
D3. The frozen HD cannot be reused to raise the bound to 0.75. A future reopen
requires an independent mechanism hypothesis, a new derivation corpus and a
new held-out corpus, or a production fix that is evaluated under an unchanged
contract with fresh anti-overfit provenance.
