# Round8 prep provenance — 2026-09-18 UTC

Evidence: CODE_INSPECTED / LOCAL_VERIFIED only, unless stated otherwise.

- Independent worktree: `/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep`.
- Branch: `codex/round8-qwen35-prep`.
- Base: `159e3dfcbbda94a13a36d004dde47c1fd9f05374` (committed Round7 snapshot, not final scheduler freeze).
- Parent gitlink: `f02f304a25364d1e8fe9c52515a22f20f5089a52`.
- Prior deployed child: `eaa58ccea4ee372b8b207dd16f398eafe3da37ff` plus Round7 recorded overlays; not interchangeable with gitlink.
- Official KT snapshot: `105359b4bf70073af9fab992ad29920666acccc9`, resolved by `git ls-remote`, fetched without checkout changes.
- Its SGLang gitlink: `3424f35d0e60f03b9e75d9b90941f8f4cb944524`, verified by `git ls-tree`.
- Official SGLang comparison snapshot: `9784d5f979c9eaa5a6d9b31e596a7a73cb76dab8`.
- Source snapshots retained under `02_OFFICIAL_AUDIT`; GitHub API returned 403, so Git transport and pinned raw URLs were used instead.

No Round7 checkout/submodule/runtime/package/container/process changed. No Qwen checkpoint shards downloaded, no NPU occupied, no conversion or remote bulk hashes. No push, merge or tag.

The known `.22` Round7 environment is historical handoff information, not a new probe: Kunpeng920/910B3, CANN9.0.0, torch2.9.0+cpu, torch_npu2.9.0.post2, Python3.11.15, CPU128–191/memnodes4,5/NPU2. Re-measure before Round8 admission. Do not reuse its writable runtime.
