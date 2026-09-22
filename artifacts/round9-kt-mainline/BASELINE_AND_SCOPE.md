# Round9 KT mainline: execution scope

Authorized by user `kaishizhixing`, 2026-09-22. Execute T0–T4, decision and T6. T5 scheduling experiment remains deferred. No push, merge, tag, timers, dependency upgrades or training modifications.

Baseline: `57de93dfbb5775f1265378c1eae737edb8d5cc3e`.
Worktree: `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline`.
Branch: `codex/round9-kt-mainline-attribution`.
Frozen remote: `/mnt/data1/kt-round9/framework-dev-20260921-p1-timeline-v6`.
Physical 15, host `110.123.0.3`, shared with exact training container `0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef`.

The v61 explicit profile is Q32 / C128 / MXFP4 CPU + original BF16 NPU / Layerwise + streaming32 / PLE ASYNC. Global Async default remains OFF. Preserve that explicit profile for comparisons. A successful one-token response does not prove a decode forward.

No kernel, bank, row algorithm, placement migration or cache authority changes. Protected qualified modules remain byte-identical. New observers default OFF. No device completion or overlap inferred from host timestamps. All missing device/worker timings remain UNKNOWN.
