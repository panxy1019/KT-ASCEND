# D1 reuse and system entry

The user requested migration of the previously implemented D1, not a new
Deferred coordinator. Inspection confirms the deployed Round7 runtime already
inherits that implementation. No redundant production rewrite was performed.

## Source continuity — CODE_INSPECTED

- Historical call graph and fixes: `migration_bridge/round6/01_DEFERRED_CALLGRAPH.md`
  and `02_DEFERRED_TOPOLOGY_AND_LIFETIME.md`.
- The deployed CPU wrapper differs from retained `/workspace/kt-src/kt-kernel/python/experts_base.py`
  only by two opt-in Round6.2 capture hooks; its Deferred execution logic is retained.
- Deployed `experts_base.py` SHA256:
  `d208f1d71afc1a8aed2554939a7c53358294b2ae49d4b1af5791de6ad93186c3`.
- Deployed SGLang `kt_ep_wrapper.py` SHA256:
  `3e5d6a38c375ed6e8d538cd4e3eea3b8dad2d725313f8554fe5a1b3e2ff0fbb7`.
- Topology-aware `resolve_layer_max_deferred`, adjacent-layer consumption,
  empty-CPU-route suppression, two-slot buffers, CPUInfer pending synchronization,
  and framework `copy_` transfers are present. Existing C64 native and FP32 merge
  fixes remain intact. Debug counterfactual replay is not a production replacement.

## Current execution

Host110.129.0.22, owned container `kt-r67-c64-development-v1`, runtime-v8,
Static Uniform R8-C64, Dynamic OFF, Deferred1, graphOFF, batch1, TP1.
Actual server PYTHONPATH confirms the frozen CPU and SGLang runtime directories.

System run: `round7/return22-deferred-system-v1`, log
`round7/return22-deferred-system-v1.log` under the existing results mount.
Three processes execute32 requests each for repetition, prefix, restart and
observer parity. Trace must prove actual deferred CPU routes and terminal-layer
budget/pending zero. Full result is pending, not a qualification PASS.

Local regression: three `test_deferred.py` tests passed, including rejection of
flag-only/no-execution evidence and terminal pending leakage.

D0 retained static evidence is the baseline. After D1 system PASS, continue
D0/D1 performance and frozen CMMLU; do not infer quality from system success.
