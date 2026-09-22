# Round9 takeover and source preflight — 2026-09-20

Status: R9-A IN_PROGRESS. This is read-only takeover/source evidence, not
resource admission, model identity PASS, or device qualification.

## Execution ownership and live transport

The new session read both 2026-09-20 handoffs, all CURRENT_EXECUTION history,
EXECUTION_PLAN and amendments 05/06. Initial local HEAD was
2aab965f7c9e00051055375114a5f38abba25ccc on
codex/round9-heterogeneous-working-set, with a clean worktree.
The user now specifies admin@110.120.0.3 → sudo -i →
admin@110.129.0.22 → sudo -i for interactive access. This was exercised.
It does not change the existing proxy tunnel or authorize deployment on .3.

At 01:34:16 UTC, .22 unit kt-round9-range-recovery-v3 was ACTIVE with original
PID2233665, MemoryMax17179869184 (16GiB), swap0, and OOM/kill0.
61/131 size-complete published shards total176125078536bytes; unpublished
committed ranges total8992587776bytes. Combined185117666312/360000192888
bytes (51.42%); published-shard progress was excluded to avoid double counting.
These counts are not whole-model SHA qualification. No verification-*.json
receipt existed. The local proxy tunnel was ACTIVE and automation round7
remained PAUSED; neither was changed.

Three deployed scripts and the manifest match local SHA256; see
evidence/takeover-20260920-0134.json. The remote manifest is directly at
/home/admin/kt-round9-qwen38/download_source.json, not under manifests/.
Local tests: python3 -m unittest discover -s migration_bridge/round9/tests,
18/18 PASS. This covers tools only.

New range failures occurred around01:33–01:34: curl exit28, approximately
89–90seconds, with29.6–31.6MB received out of33554432bytes. Source sets
--max-time90 and preserves failed parts before bounded retry. This establishes
deadline expiry before full range receipt; it does not establish the underlying
network slowdown's cause. At01:35, shards66/67 advanced to671088640 and68 to
536870912 via the existing retry path. Keep the progressing controller intact.
Do not report an uninterrupted healthy run or claim a transport fix. No timeout,
memory, worker, proxy, or system service settings changed during takeover.

Host inventory was observational: about2.5TiB disk available,
MemAvailable1035498164KiB; NPU0–7 report910B3/healthOK and no running NPU
processes in npu-smi25.5.1. Unrelated mindspeed-mm and verl containers remained
running and untouched. These snapshots do not reserve CPUs/NPU or approve
loading peaks. C64/C128 resource admission remains pending.

## Recovered architecture source, independently inspected

An existing clean audit checkout at /tmp/sglang-qwen4-audit-20260914 has
origin https://github.com/sgl-project/sglang.git and HEAD
4ccff141dbe992794f9da6c3aa23535b4f72000d. Fifteen relevant source files match
their committed Git blobs byte-for-byte. Their SHA256/Git blob identities are
in evidence/source-preflight-20260920.json. Treat this as a candidate semantic
source; its older audit results and device qualifications are not inherited.
No checkout, merge, cherry-pick or runtime installation was performed.

Separately, committed kvcache-ai/sglang pin
541ddc37cbc92c60dc748db5ff1a2aad0b069a80 was inspected through the object
database in the existing kt-main-integration submodule. Entire python/sglang
tree searches for Qwen4Exp, qwen4_exp, ple_layer_ids and hc_lowrank returned no
matches (exit1). Registry imports module EntryClass and permits a Transformers
fallback/external package. Therefore native Qwen4 registration is not present
in that committed tree; this does not prove every external/fallback path is
unsupported. Its dirty untracked config/hyperconnection files were not reused
or modified. The current Round9 gitlink is separately f02f304a25364d1e8fe9c52515a22f20f5089a52.

Official KT f7607c0f66c643220ff43b822b24483c98c41392 .gitmodules resolves
third_party/sglang to kvcache-ai/sglang. Its exact gitlink is still PENDING:
GitHub tree API requests returned403, and the KT commit is not in the local
object database. Do not infer a gitlink from a remote HEAD observation.

## Source-level findings and remaining proof obligations

All line references below are to the clean 4ccff141 source snapshot.

| Contract | Inspected source | Result / remaining gate |
|---|---|---|
| PLE numbering | models/qwen4_exp.py:1291–1295 uses layer_id+1 | Config[2] maps to zero-based layer1; do not patch this as an off-by-one error. Verify real checkpoint/table identity later. |
| Packed experts | qwen4_exp.py:2085 splits gate/up with chunk(2,dim=-2);1875–1893 iterates loaded_weight[expert_id] | Source expects expert axis0, gate then up on penultimate axis. Actual shape/dtype/header and every conversion still require proof. |
| Registration | qwen4_exp.py:2182 EntryClass includes Qwen4ExpForConditionalGeneration | Present in this upstream snapshot, absent by inspected names in committed kvcache pin. No Ascend import test yet. |
| Gated residual | layers/hyperconnection.py:193–225 | Read uses SiLU(linear(normed)/hc), second linear, sigmoid elementwise and branch mean. Write adds block output weighted by2*sigmoid(linear(normed residual)/hc). Preserve division byhc and all branches; equations are not device PASS. |
| PLE ID layout | qwen4_exp.py:448–574 | ngram_heads=(ngram_size-1)*heads_per_ngram, deterministic splitmix multipliers, prime per-head vocab sizes and offsets. Full BOS/EOS/history/chunk/row audit still pending. |
| Port boundary | qwen4_exp.py imports Triton, elementwise kernel, symmetric allocator and Qwen3.5 components | Cannot copy the full file and assume ARM/Ascend support; separate semantic reference from backend dispatch. |
| CPU precision | local kt-kernel/operators/llamafile/moe.hpp:116 onwards | aarch64 without SVE maps BF16 weight effective vec-dot input to F32. Shape640/320, actual extension, grouped-prefill and C64 execution remain NOT_RUN. |

## Next gates

1. Let the existing downloader finish and inspect its one final manifest audit
   plus tensor census. Preserve all failures; never launch a parallel full SHA
   audit or a duplicate downloader.
2. Reconcile main/PLE/MTP/vision tensor identities and capacity, then freeze
   independent runtime/resource admission and the minimal semantic port.
3. Execute bounded F32 canonical vs BF16-storage/F32-compute probes first;
   select reference only after C64 system/shape qualification. No450GiB full
   F32 bank by default. MXFP4 then Q8_0 remain independent candidates.
4. Advance the ordered full-model/quality gates, then same-precision C128
   (4×32 pools,160×4) with independent admission. No previous model PASS applies.

Round8 performance remains cancelled; no push, main merge, tag, old heartbeat
resume, or new scheduled continuation was performed. Download completion and
later device gates are still outstanding.
