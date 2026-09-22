# Original-weight shared expert and final mixer gate

2026-09-20 physical15, active shared-training environment. Runtime model MoE
class is f02 Qwen2MoeSparseMoeBlock (also used by Qwen3.5), not an invented
Qwen3NextSparseMoeBlock. Config/source have no extra routed scaling factor.

Original layer0 router/shared gate/gate-up/down weights and final GR mixer's
three weights were loaded through real parameter loaders, with private CPU
clones before transfer. Tokens1/7/17, three repeats each, all hashes stable.

Independent CPU FP64 relative errors:

| Component | Maximum relative L2 | Gate |
|---|---:|---:|
| Shared expert including sigmoid gate | 0.003950924123 | 0.01 |
| Router logits | 0.001665434032 | 0.01 |
| Final mixer output | 0.003913529232 | 0.01 |
| Final mixer internal branch norm | 0.001685349653 | 0.01 |

Actual MoE ordinary forward was tested with an identifiable routed-output
fixture: routed called once and result exactly equals one BF16 routed+shared
addition. This boundary result is NOT routed computation, TopK or KTEP evidence.
Report28 provides the separate actual KTEP CPU/NPU evidence. A complete real
single decoder is the next gate.

Final mixer formula independently checks per-branch RMS, Gemma1+w, lowrank
projection/4+SiLU, sigmoid gates and mean across four branches. No checkpoint
final-norm tensor exists. This qualifies the component, not whole-model logits.

Evidence: evidence/physical15/20260920-shared-mixer/shared-mixer-v1.json and log,
exact probe, container config/state and SHA256SUMS. Container exited0/OOMfalse.
Full model remains NOT_RUN; timers PAUSED; no push/main/tag.
