# Round 2A Git Commits

## Repository state

- base：`c40d37c63cb9b7d041c8489167b3b822961d12c5`
- branch：`feature/kt-arm-llamafile-cpu-plane`
- policy：每个 production defect 独立 fix commit；不 squash；不纳入 Round 1/1.5 既有 untracked 文档

## Commits

1. `734da50` — `fix(cpu): report native ARM variant on aarch64`
   - ARM build/runtime/extension metadata；x86 regression tests。

2. `aba93a3` — `test(llamafile): add deterministic GGUF MoE fixture`
   - local deterministic generator、manifest、byte reproducibility、wrapper E2E test。

3. `4f9a5cc` — `fix(llamafile): support CPU wrapper execution and expert mapping`
   - pure CPU forward、strict mapping contract、C++ physical→logical placement、path-aware loader reuse。

4. `ea61b53` — `test(moe): cover ARM routed expert and prefill correctness`
   - 4/8 experts top2、mapping、qlen/edge route tests。

5. `7377568` — `test(cpuinfer): harden LLAMAFILE lifecycle and NUMA coverage`
   - 1000 loop、20 lifecycle、two layers、threadpool/NUMA diagnostics；config-keyed CPUInfer reuse。

6. `e2af3ee` — `fix(cpu): allow wrappers without a pinned allocator`
   - A3 CPU-only E2E 揭示的 narrow allocator fallback。

7. `0cfb771` — `fix(llamafile): enforce the prefill hidden-size contract`
   - A3 qlen 32/64 揭示的 grouped-prefill QK_K hidden alignment contract。

8. `4fe8668` — `test(moe): apply BF16 numerical acceptance thresholds`
   - 用 max abs / mean abs / relative L2 替代 bit-exact zero 判据。

9. documentation commit — 本目录 11 份审计、测试、风险和交付文档（最终 hash 见 `ROUND2A_SUMMARY.md` 或 `git log`）。

## Why more than the recommended 4–6

任务的 4–6 是推荐目标，同时明确要求 discovered production bug 单独 fix commit。A3 实测发现两个独立 production 问题（CPU-only pinned allocation、prefill hidden alignment）和一个独立 numerical test-policy correction；保留分离提交使每个变更可审查、可回退，并避免把测试与 defect fix 混在一起。
