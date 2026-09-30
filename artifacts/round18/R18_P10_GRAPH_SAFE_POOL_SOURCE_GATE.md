# Round18 P10 Native expert pool graph-safe residency source gate

Status: `SOURCE_GATE_ONLY_NOT_NPU_PASS`

Disposition at source-audit time: `PROCEED_TO_BOUNDED_NPU_COMPONENT_PROBE`

Terminal resolution: this recommendation was superseded by the fresh route
economics and graph-compatibility gate. Round18 stopped the device probe because
the only capacity-safe pool still lacked a current-token device planner. The
source seam remains a future entry point; see `R18_PHASE_II_POOL_ECONOMICS.json`,
`R18_EXPERT_POOL_GRAPH_COMPAT.json` and `FINAL_STATUS.json`.

This audit is read-only with respect to the frozen vLLM/vLLM-Ascend source and did not start an NPU, model, container, or remote task. It does not qualify `FULL`, `FULL_DECODE_ONLY`, `PIECEWISE`, weight replacement, or mapping publication on device.

## Finding

The frozen stack contains a concrete adjacent mutation seam that is narrower than a general expert-pool implementation but strong enough to justify one cheap device component probe.

`ACLGraphWrapper` captures by calling the Python runnable once, then replays with only `entry.aclgraph.replay()`. For FULL it synchronizes the current stream immediately before replay. The model runner calls Dynamic EPLB `forward_before()` before `_model_forward()` and `forward_end()` after it, so EPLB planning and publication are outside the captured full-model runnable. The EPLB loader receives weights into separate buffers, waits for transfer requests, overwrites existing expert tensors with `expert_tensor.copy_(buffer_tensor)`, and updates the existing NPU `log2phy` tensor with `copy_`. Native MoE forwards that same `log2phy` tensor to the quant method.

This is source evidence for fixed-address weight and mapping mutation around replay. It is not evidence that an already captured ACLGraph reads the new contents correctly on this NPU/CANN stack; that is the exact fact the device component probe must measure.

## Exact boundary

The current Round18 CPU/Python planner remains incompatible with FULL/FULL_DECODE_ONLY replay because replay does not re-enter `apply()` or any per-token Python callback. The frozen source also has no route-miss primitive that exposes the current token's selected experts before the same replay without Host readback. Its adjacent Dynamic EPLB path uses accumulated historical load, a CPU subprocess, and graph-external updates for a later forward. It is therefore evidence for a publish mechanism, not a current-token miss scheduler.

The existing EPLB implementation cannot be reused as-is for the Round18 EP1/Q32 pool. `init_eplb_config()` explicitly rejects dynamic EPLB when `ep_size == 1`; its loader is D2D-based; and the audited source does not prove that Round18's Q32 global-to-slot map is interchangeable with EPLB `log2phy`. The loader also publishes `log2phy` before enqueuing the final fixed-slot weight copies. Python/stream ordering may make this safe before the next forward, but atomic weight-then-map visibility is not proved by source inspection.

These gaps do not imply that every Native pool design is permanently invalid. A static or prior-token residency plan, a graph-external prefetch/publish protocol, a device-resident miss mechanism, or a two-phase route/execute graph can avoid the current CPU seam. Each would need separate evidence and economic qualification.

## Mode-specific assessment

| Mode | Frozen source result | What remains |
|---|---|---|
| FULL / FULL_DECODE_ONLY | Graph-external mutation seam exists; Python planner inside model forward is skipped on replay | Prove stable addresses, completed copy ordering, mapping publication, and new-weight visibility in one captured Native MoE replay |
| PIECEWISE | The same outer EPLB hook brackets the model call; individual pieces replay while Python model orchestration still runs | Prove that the selected Native MoE piece observes same-address weight/map mutation; current Round18 full-model PIECEWISE remains independently unqualified |
| Weight copy inside ACLGraph capture | Not established | Not required for the minimum probe; perform copy and publish outside capture between replays |

## Minimum real-NPU cheap probe

1. Instantiate one frozen native unquantized MoE layer with two preallocated physical slots. Avoid the full model and do not enable the existing EP-only Dynamic EPLB planner.
2. Record the weight and mapping `data_ptr` values. Capture a legal graph invocation in the mode under test with logical expert A mapped to the target slot, and validate the first replay against an eager A reference.
3. Between replays, outside capture, copy expert B into the same fixed weight storage. Wait for copy completion, then update the same mapping storage without replacing either tensor object.
4. Replay without executing Python model forward and without reading selected routes back to Host. Compare with eager B, stale-A, and unchanged-slot controls. Run the reverse B-to-A transition to catch one-way cache effects.
5. Record resolved graph mode, capture/replay counts, pointer identities before and after, stream/event ordering, output hashes and numerical errors, allocator deltas, exceptions, and cleanup. Only this run can promote weight/map visibility to PASS.

For PIECEWISE, repeat the same two-state transition with the Native MoE piece captured. Do not use success there to claim the existing Round18 Qwen/GDN PIECEWISE full-model path is qualified. For FULL_DECODE_ONLY, first prove the component in FULL; dispatcher eligibility and request-level correctness are later gates.

## Reproduction

```bash
cd /home/admin/Documents/ChatGPT/KT/kt-round18-native-preserving
python3 -m unittest tests.round18.test_expert_pool_graph_contract
python3 -m round18.expert_pool_graph_contract \
  --source-root /home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round15/upstream-image-source/round15-offline-vllm-source-20260928
```

The machine-readable result is `R18_P10_GRAPH_SAFE_POOL_SOURCE_GATE.json`. The audit module verifies the frozen files are present, records their SHA-256 identities, and closes the source gate when any required wait/copy/publish/consumer primitive is absent.
