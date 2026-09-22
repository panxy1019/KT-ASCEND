# Round 6.7 — Serial All-MoE Resident-Ratio Sweep

User-authorized scope, 2026-09-17. Audit base: `93844959b32f64a37b70346bd5c7e0df2394f5d4`.
Branch: `codex/round6-7-serial-all-moe-ratio-sweep`.

This is a **new product screening experiment**, not a repair or retrospective
approval of the Round6.5 numerical contract. Round6.5/6.6 strict failures and
Round6.6B zero-observed-quality-loss failure remain historical failures.

The latest user explicitly accepts **CMMLU-only, 5pp** product screening. No
GSM8K gate is inferred. Product acceptance means system PASS and an observed
CMMLU256 accuracy delta >= -5pp; it does not prove population non-inferiority,
strict numerical equivalence, or production readiness.

## Frozen execution

- One existing container: `kt-r65-static-offload`, host `110.129.0.22`.
- Ascend **910B3**, NPU0 only. No new containers or business workload changes.
- Serial R64 → R32 → R16 → R8, all 26 MoE layers (1–26).
- TP1, batch1, 32 CPU workers, one threadpool, NUMA0, CPU0–31.
- Deferred / Dynamic / Graph / MTP / Speculative OFF; production overlap.
- CPU F32 LLAMAFILE, NPU BF16/unquantized. Zero production code changes.
- Deployed parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7`, child
  `eaa58ccea4ee372b8b207dd16f398eafe3da37ff`, plus recorded overlays.
- Runtime source manifest SHA256
  `68d1a5814a70e123eb55c840ec9a995fda1fe4802120374f61f32e2d1775570a`.
- Model revision `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Tokenizer SHA256 `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`.
- GGUF SHA256 `20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2`,
  57,579,411,648 bytes. Shared existing file; no conversion or duplicate copy.

Local `third_party/llama.cpp` was already dirty before this round; preserved,
not used as a new native build. CPU NUMA0 is **not local** to NPU0 (NUMA6).
Performance is comparable only within this fixed topology, not an optimized
NUMA deployment claim. Exact imported-file and asset hashes are checked before
and after every model session.

## Evidence vocabulary

CODE_INSPECTED / LOCAL_VERIFIED / TARGET_VERIFIED are distinguished. This
server is 910B3; historical A3 evidence is not relabeled as new execution here.
No profile can be accepted merely because an automated job was launched.
