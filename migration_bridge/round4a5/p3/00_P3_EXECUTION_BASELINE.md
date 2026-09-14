# P3 Execution Baseline

Evidence state: `A3_VERIFIED`

## Immutable lineage

| Item | Value |
|---|---|
| P2 immutable tag | `round4a5-p2-a3-verified` |
| P2 tag target | `d4830525c6806d5d587b3096634614aad7fef181` |
| P3 branch at execution start | `feature/kt-round4a5-p3` |
| local P3 documentation base | `25821cd22d6a1c63f43467a9f0e07bf5958182b3` |
| A3 mounted source HEAD | `4e3a488962778bed5aacc260c25becef222e973d` |
| SGLang child | `cbaa79c2f5b004ab4ca470c9fa56161666ccafb2` |

The A3 mounted checkout predates the final P2 evidence-only commits. Static
comparison proves that `4e3a488...` and the immutable P2 tag target have zero
differences under `ktransformers`, `kt-kernel/src`, `kt-kernel/include`, and
`setup.py`. The only tracked A3 worktree changes were the two retained
test-contract corrections. Untracked `kernel_meta/` and `fusion_result.json`
were framework-generated runtime artifacts. No production source was changed
during P3.

No upstream KTransformers merge, rebase, or source substitution occurred.

## A3 environment

| Item | Value |
|---|---|
| environment ID | `a3-r4a5-p3-0825d8eb90af` |
| host | `a3-server-00`, aarch64 |
| CPU | Kunpeng 920 7280Z, 640 logical CPUs, 8 NUMA nodes |
| container | `kt-r4a5-p2-closure-env` |
| container ID | `0825d8eb90afcba292433acc9230780c575f86a5853b34db01d1141ee36ff209` |
| image | `sha256:7781588ef508d2cf37bb4a09878b0079bf13795fedd874cd7c76c4da47f2ea3a` |
| CPU affinity | cpuset `0-15` |
| accelerator visibility | `/dev/davinci0` only |
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| Python | `3.11.15` |
| torch | `2.9.0+cpu` |
| torch_npu | `2.9.0.post2` |
| Transformers | `5.3.0.dev0` |
| tokenizers | `0.22.2` |
| host modified | NO |
| business containers modified | NO |

All serving processes were stopped after qualification. The final NPU check
showed no process and zero AICore use on the assigned NPU0.

## Frozen runtime

`TP=1`, batch 1, greedy decoding, production overlap, graph/deferred/dynamic
placement/MTP/speculative decoding all OFF. CPU experts use LLAMAFILE/CPUInfer;
NPU experts use the frozen Ascend BF16 unquantized path. The framework-managed
Tensor `copy_` transfer fix remained active.

P3 execution ran from 2026-09-12 00:14 UTC through 06:02 UTC. The initial
pairwise instrumentation attempt lacked the required `dump.arm` sentinel and
was preserved as `INSTRUMENTATION_SETUP_FAILURE`; adding the sentinel changed
only debug capture activation, not model execution semantics. The successful
pairwise artifacts are under the canonical `p3-pairwise` evidence root.
