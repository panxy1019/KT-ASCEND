# QD2 toolchain deployment supplement

Read-only SHA256 comparison during the healthy defer0 sweep on 2026-09-15:
all seven files below are byte-identical between the local committed source
and A3 `/workspace/results/round6_2-prequalification/tools/` (the latter is
bound to `/home/admin/kt-artifacts/round6-deferred/round6_2-prequalification/tools/`).
This supplements rather than rewrites the captured per-forward provenance.
It records observed file identity; it is not a numerical or model gate.

| File | SHA256 |
| --- | --- |
| run_qualification_capture.py | f2845e73175838212d7fe3ee81332b0db43d1a479d088dd1c5d0d2fb17356cde |
| assemble_capture.py | eab020e03e6414dbfc4bcb7ad81d451e62374416a86db1e6ea53949fb4db273a |
| validate_position.py | e80f194e4e0db9529168eff15e02873430a597efd5c59c0718be0d0500eb9950 |
| run_owned_process.py | c7b4e23cf71a94e3a9d379a6a84ffa324fd3bf032389583c1501e06ee2a89dea |
| evaluate_contract_v2.py | ecb5c8f758b62fec1bd25d738dc0a654462188f88c3b8d3a49183fe5c659a3d8 |
| tail_statistics.py | 1d5381a8528fed80a34bd5f63e886502bcab32f6dbacc6eb61353585a9a18c68 |
| audit_deferred_transitions.py | c8842c151246e20e9ef36dc50fc4ea7ed88fbf74f1877306b0210e751a262732 |

The four live collection/guard modules were not changed. The offline
numerical evaluator's additional transition-audit preflight was deployed
before any QD2 numerical comparison. The registered metrics, rank rule,
confidence, candidate K and exposure minimum were not changed.

The prepared defer1 run metadata and the active defer0 metadata were compared
as complete parsed JSON objects: the sole differing key is `mode`. All source,
binary, GGUF, placement, model/tokenizer, history and corpus hashes are equal.
Mode metadata files and their raw hashes are retained under `evidence/`.
