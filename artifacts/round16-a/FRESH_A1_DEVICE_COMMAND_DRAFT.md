# Round16-A fresh A1 future device command draft

Status: `COMMANDS_FROZEN_READY_AFTER_NEW_FRESH_ADMISSION_DEVICE_NOT_RUN`

None of these commands ran in the offline task. The builder, real single-process six-request driver, host supervisor, copy-on-spend ledger, guard cleanup, archive writer, and strict Round13/A1 evidence normalizer are implemented. The copied snapshot binds the verified Round13 observer to `/out/round13-spans.jsonl` and `/out/round13-bindings.json`; the custom observer records Host data for layers 0..47. The normalizer copies direct Round13 QSA/GDN/NPU-resident events, verifies all-layer Host coverage, records a serial scheduler Decode bounding interval, and explicitly returns `INSUFFICIENT_ATTRIBUTION` when no direct isolated A2/A3 interval exists. It never creates a candidate ratio from nested-span arithmetic.

## Frozen inputs

- Parent: `/home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14a/a-gdn-model-snapshots/v295`
- Parent policy: read-only; a build rejects an output inside the parent tree and patches only a fresh copy.
- Block A: `round16-a1-fresh-profile-a-v001`
- Block B: `round16-a1-fresh-profile-b-v001`
- Initial ledger: `artifacts/round16-a/FRESH_A1_LEDGER_INITIAL.json`
- Total budget: 12 attempts, exactly six per block, zero retries, zero abort controls.

The fixed per-block queue is:

1. W1 warmup
2. W2 warmup
3. W1 measurement
4. W2 measurement
5. short drain using `canary_math391`, 27 input tokens, four new tokens, `ignore_eos=true`
6. fixed quality canary using the same frozen case, normal EOS, `max_new_tokens=32`, temperature 0, top-p 1, seed 20260920, exact output `391`, and finite logprob

The fixture is `v295/round10-t1-fixture.json`, SHA-256 `21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95`. Its `canary_math391` input ID SHA-256 is `4d2584778859d8b34fa388d6fb09925c1da1a0197945de1298f09d199dce489d` and messages SHA-256 is `e5a27eaad7efa96e5ff9020ca99f901cf968238ac7ae7b662c7369923c5e6615`.

## 1. Build fresh copies

Run from the Round16-A worktree only after replacing the fresh output paths. `build_snapshot` checks all eleven frozen source, helper, fixture, budget, and monitor hashes before creating output, copies the tree, patches the copy, compiles patched Python, writes `round16-a1-block.json`, and writes `round16-a1-source-sha256.json`.

```bash
python3 -c 'import sys; from round16.a.fresh_a1_snapshot import build_snapshot; build_snapshot(sys.argv[1], sys.argv[2], block=sys.argv[3])' \
  /home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14a/a-gdn-model-snapshots/v295 \
  <FRESH_A_SNAPSHOT_PATH> A

python3 -c 'import sys; from round16.a.fresh_a1_snapshot import build_snapshot; build_snapshot(sys.argv[1], sys.argv[2], block=sys.argv[3])' \
  /home/admin/Documents/ChatGPT/KT/kt-round9-mainline/artifacts/round14a/a-gdn-model-snapshots/v295 \
  <FRESH_B_SNAPSHOT_PATH> B
```

The copied launcher still defaults the observer to `ROUND16_A1_PROFILE_ENABLE=0`. Enabling requires the later reviewed controller input. These two complete snapshots must be built on the local host from this trusted read-only donor, then transported as whole directories. After each build, freeze the SHA-256 of `round16-a1-source-sha256.json`, the initial `execution-budget.json` SHA reported by offline preflight, and the initial ledger SHA in a reviewed transport receipt. The remote host may compare against those fixed values; it must not calculate a new expected value and authorize itself.

The actual remote v295 parent is `/mnt/data1/kt-round9/round14a-a-gdn-model-20260927-v2/v295`. Its ten immutable non-budget inputs match the frozen identity, but its historical `execution-budget.json` SHA-256 has prefix `046fc30c`, whereas the required trusted initial budget is `665ded641f8848d65ca44776cf74dfd70909184339a619f3d70befee09c36974`. Therefore that remote parent is identity evidence only and is rejected as a build donor. Do not edit, reset, or overwrite it.

## 2. Run offline preflight

For A, use a fresh copy of the initial zero-use ledger. For B, use A's `round16-a1-ledger-after.json`, which must show A=6 and B=0.

```bash
python3 -c 'import json,sys; from round16.a.fresh_a1_snapshot import offline_preflight; print(json.dumps(offline_preflight(sys.argv[1], block_id=sys.argv[2], ledger_path=sys.argv[3]), indent=2))' \
  <FRESH_A_SNAPSHOT_PATH> round16-a1-fresh-profile-a-v001 \
  <FRESH_SHARED_LEDGER_BEFORE_A.json> > <FRESH_A_PREFLIGHT.json>

python3 -c 'import json,sys; from round16.a.fresh_a1_snapshot import offline_preflight; print(json.dumps(offline_preflight(sys.argv[1], block_id=sys.argv[2], ledger_path=sys.argv[3]), indent=2))' \
  <FRESH_B_SNAPSHOT_PATH> round16-a1-fresh-profile-b-v001 \
  <FRESH_A_OUTPUT>/round16-a1-ledger-after.json > <FRESH_B_PREFLIGHT.json>
```

Preflight is read-only and performs no device probe. It rejects source-manifest drift, block mismatch, nonzero current-block use, an incoherent 12-attempt ledger, default enablement, or a raw receipt already present in the snapshot. `execution-budget.json` is deliberately outside the immutable source-file set: its separate contract permits only `INITIAL_UNSPENT` or one launcher-written `SINGLE_MATCHING_START` with the exact block container name and a full 64-hex container ID.

## 3. Prepare and review controller inputs

`prepare_controller_input` requires exact enablement, block identity, ledger byte SHA-256, source manifest, and a path that does not yet exist. It returns a plan and does not start a process.

```python
from hashlib import sha256
from pathlib import Path
from round16.a.fresh_a1_snapshot import prepare_controller_input

ledger = Path("<FRESH_SHARED_LEDGER_BEFORE_A.json>")
plan_a = prepare_controller_input(
    "<FRESH_A_SNAPSHOT_PATH>",
    block_id="round16-a1-fresh-profile-a-v001",
    ledger_path=ledger,
    expected_ledger_sha256=sha256(ledger.read_bytes()).hexdigest(),
    output_dir="<FRESH_A_OUTPUT>",
    enabled=True,
)
```

Repeat for B with its block identity and A's ledger-after file. Compare the filled plans with `FRESH_A1_CONTROLLER_INPUT_A.json` and `FRESH_A1_CONTROLLER_INPUT_B.json`; those checked-in templates remain disabled and non-executable.

## 4. Frozen remote paths and launch commands

Stage the reviewed A control tree read-only at `/mnt/data1/kt-round9/artifacts/round16-a/control-v001`. Transport the two already-built local snapshots and the reviewed initial ledger to the new paths below. No remote `build_snapshot` command is legal because the actual remote parent has a spent historical budget. The transport step is environment-owned and is intentionally not executed by this offline task.

```bash
export R16A_CONTROL=/mnt/data1/kt-round9/artifacts/round16-a/control-v001
export REMOTE_V295_IDENTITY_ONLY=/mnt/data1/kt-round9/round14a-a-gdn-model-20260927-v2/v295
export A_SNAPSHOT=/mnt/data1/kt-round9/artifacts/round16-a/a-snapshots/fresh-a-v001
export B_SNAPSHOT=/mnt/data1/kt-round9/artifacts/round16-a/a-snapshots/fresh-b-v001
export LEDGER0=/mnt/data1/kt-round9/artifacts/round16-a/FRESH_A1_LEDGER_INITIAL.json
export A_SOURCE_MANIFEST_SHA256='<REVIEWED_LOCAL_A_MANIFEST_SHA256>'
export B_SOURCE_MANIFEST_SHA256='<REVIEWED_LOCAL_B_MANIFEST_SHA256>'
export LEDGER0_SHA256='<REVIEWED_LOCAL_INITIAL_LEDGER_SHA256>'
test "$(sha256sum "$A_SNAPSHOT/round16-a1-source-sha256.json" | cut -d' ' -f1)" = "$A_SOURCE_MANIFEST_SHA256"
test "$(sha256sum "$B_SNAPSHOT/round16-a1-source-sha256.json" | cut -d' ' -f1)" = "$B_SOURCE_MANIFEST_SHA256"
test "$(sha256sum "$LEDGER0" | cut -d' ' -f1)" = "$LEDGER0_SHA256"
PYTHONPATH="$A_SNAPSHOT" python3 -c 'import json,sys; from round16.a.fresh_a1_snapshot import offline_preflight; r=offline_preflight(sys.argv[1],block_id=sys.argv[2],ledger_path=sys.argv[3]); assert r["source_manifest_sha256"]==sys.argv[4] and r["execution_budget_state"]=="INITIAL_UNSPENT"; print(json.dumps(r,sort_keys=True))' "$A_SNAPSHOT" round16-a1-fresh-profile-a-v001 "$LEDGER0" "$A_SOURCE_MANIFEST_SHA256"
PYTHONPATH="$B_SNAPSHOT" python3 -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); manifest=root/"round16-a1-source-sha256.json"; m=json.loads(manifest.read_text()); assert h(manifest)==sys.argv[2] and m["block_id"]=="round16-a1-fresh-profile-b-v001"; assert all(h(root/name)==value for name,value in m["files"].items()); assert h(root/m["execution_budget_contract"]["path"])==m["execution_budget_contract"]["initial_sha256"]; print(json.dumps({"status":"PASS_TRANSPORT_SOURCE_ONLY_B_A_LEDGER_PENDING","source_manifest_sha256":h(manifest),"execution_budget_state":"INITIAL_UNSPENT"},sort_keys=True))' "$B_SNAPSHOT" "$B_SOURCE_MANIFEST_SHA256"
```

The B transport check is intentionally source-only. A full B `offline_preflight` is invalid until A has completed, because B requires the exact A=6/B=0 predecessor ledger and frozen A request IDs. Do not synthesize a B predecessor. The real B host launch runs the complete preflight against A's archived ledger-after and cleanup/archive evidence.

Immediately before each block, the integration admission runner must create a new launcher-specific two-round receipt containing `status=PASS`, `started_at`, `ended_at`, `validated_at_epoch`, the exact physical12 target, two unique basename-only entries in `rounds`, and the two raw round hashes in the same order. For the next run, use `RESUME02_ADMISSION_ROUND1.json` and `RESUME02_ADMISSION_ROUND2.json`. Both raw files must be beside the result. Each raw round must last at least 60 seconds, round 2 must follow round 1, the result envelope must equal the two raw boundaries, and both `ended_at` and `validated_at_epoch` must remain at most 600 seconds old. `round15_admission.py` is the accepted two-round sampler/validator source (SHA-256 `571f51613242cd00a30cfb580a759b8fb681d2f4f1cf4cbce7d0048383e00628`); the launcher-specific wrapper must add the target, ordered `rounds`, result envelope, and `validated_at_epoch` fields before review.

For A, run the fresh RESUME02 admission and stop for external SHA review:

```bash
export ADMISSION_RUNNER=/mnt/data1/kt-round9/artifacts/round15/round15_admission.py
test "$(sha256sum "$ADMISSION_RUNNER" | cut -d' ' -f1)" = 571f51613242cd00a30cfb580a759b8fb681d2f4f1cf4cbce7d0048383e00628
export A_ADMISSION_DIR=/mnt/data1/kt-round9/evidence/round16-a1-admission-a-v001
test ! -e "$A_ADMISSION_DIR"
mkdir "$A_ADMISSION_DIR"
python3 "$ADMISSION_RUNNER" sample --output "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND1.json"
python3 "$ADMISSION_RUNNER" sample --output "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND2.json"
python3 "$ADMISSION_RUNNER" validate \
  --discovery "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND1.json" \
  --validation "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND2.json" \
  --output "$A_ADMISSION_DIR/base-validation.json"
python3 -c 'import hashlib,json,pathlib,sys,time; base,r1,r2,out=map(pathlib.Path,sys.argv[1:]); assert not out.exists(); one,two=json.loads(r1.read_text()),json.loads(r2.read_text()); value=json.loads(base.read_text()); value.update(schema=1,status="PASS",started_at=one["started_at"],ended_at=two["ended_at"],validated_at_epoch=time.time(),target={"physical_id":12,"card_id":6,"chip_id":0,"pci_bdf":"0000:85:00.0","numa_nodes":[2,3,4,7]},rounds=[r1.name,r2.name],raw_sha256={r1.name:hashlib.sha256(r1.read_bytes()).hexdigest(),r2.name:hashlib.sha256(r2.read_bytes()).hexdigest()}); out.write_text(json.dumps(value,indent=2)+"\n")' "$A_ADMISSION_DIR/base-validation.json" "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND1.json" "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND2.json" "$A_ADMISSION_DIR/result.json"
sha256sum "$A_ADMISSION_DIR/result.json" "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND1.json" "$A_ADMISSION_DIR/RESUME02_ADMISSION_ROUND2.json"
```

Repeat the same sequence after A cleanup for B with the fresh directory `/mnt/data1/kt-round9/evidence/round16-a1-admission-b-v001`; do not reuse A's samples or result SHA.

After a new fresh launcher-specific admission receipt is reviewed, run A once:

```bash
export A_ADMISSION=/mnt/data1/kt-round9/evidence/round16-a1-admission-a-v001/result.json
export A_ADMISSION_SHA256='<REVIEWED_RESUME02_A_RESULT_SHA256>'
PYTHONPATH="$A_SNAPSHOT" python3 "$A_SNAPSHOT/tools/round16_a1_remote_block.py" \
  --block A --snapshot "$A_SNAPSHOT" \
  --source-manifest-sha256 "$A_SOURCE_MANIFEST_SHA256" \
  --admission "$A_ADMISSION" --admission-sha256 "$A_ADMISSION_SHA256" \
  --ledger "$LEDGER0" --ledger-sha256 "$LEDGER0_SHA256"
```

Run B only after A cleanup and a second fresh admission:

```bash
export A_LEDGER_AFTER=/mnt/data1/kt-round9/evidence/20260928-round16-a1-fresh-profile-a-v001/round16-a1-controller/round16-a1-ledger-after.json
export B_ADMISSION=/mnt/data1/kt-round9/evidence/round16-a1-admission-b-v001/result.json
export B_ADMISSION_SHA256='<REVIEWED_RESUME02_B_RESULT_SHA256>'
export A_RAW_RECEIPT=/mnt/data1/kt-round9/evidence/20260928-round16-a1-fresh-profile-a-v001/round16-a1-controller/round16-a1-raw-receipt.json
export A_LEDGER_AFTER_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ledger_sha256_after"])' "$A_RAW_RECEIPT")"
test "$(sha256sum "$A_LEDGER_AFTER" | cut -d' ' -f1)" = "$A_LEDGER_AFTER_SHA256"
PYTHONPATH="$B_SNAPSHOT" python3 "$B_SNAPSHOT/tools/round16_a1_remote_block.py" \
  --block B --snapshot "$B_SNAPSHOT" \
  --source-manifest-sha256 "$B_SOURCE_MANIFEST_SHA256" \
  --admission "$B_ADMISSION" --admission-sha256 "$B_ADMISSION_SHA256" \
  --ledger "$A_LEDGER_AFTER" --ledger-sha256 "$A_LEDGER_AFTER_SHA256"
```

Before either launcher call, the host controller verifies the fixed locally reviewed source-manifest SHA, every tracked transported file, the reviewed ledger SHA, and the `INITIAL_UNSPENT` model-start budget. The launcher may perform exactly one transition to `SINGLE_MATCHING_START`; the host records both budget SHAs, and the container driver receives and rechecks the same reviewed source-manifest SHA. Block B additionally binds the exact six ordered A request IDs, the supplied A ledger SHA, A's PASS supervisor cleanup terminal state, and the final A archive hashes before any B subprocess. Each host controller then waits for port 30091 readiness, starts one six-request container driver, and retires the owned container on success or failure. Running/terminal cleanup binds the full container ID to the guard ACK. Pre-start `created` cleanup additionally requires container PID zero and terminates, or confirms prior exit of, the recorded bound guard before safe removal. A failed request remains spent; no automatic retry is permitted.

The A1 observer records Host dispatch, Host sync, Host merge, and the routed-expert parent across layers 0..47 with explicit span, parent, stream, event, and correlation identities. Direct QSA, GDN, and sampled resident-expert NPU events come from the frozen Round13 observer. The broader hook map in the profile contract is a boundary catalog, not an installation claim; A1 does not install an `NPU_SHARED` hook. Neither observer adds synchronization. CPU Expert remains `UNKNOWN_NATIVE_ABI_NO_CALLBACK` with null duration until a separately reviewed native callback exists.

After cleanup, the host controller combines the A1 Host48 rows with the already-qualified Round13 QSA/GDN/resident-expert direct event rows, writes `round16-a1-observer.jsonl` plus SHA-bound metadata, and runs a single-block strict comparator preflight. Direct device coverage is required only for W1/W2 measurement requests. A serialized `SHORT_DRAIN` Round13 batch after W2 is the FIFO flush witness; final-canary device completion is not required. Missing or drifted coverage fails the block before archive. If direct A2/A3 critical intervals remain unavailable, two complete blocks resolve to `A1_INSUFFICIENT_ATTRIBUTION` with `selected_candidate=null`; A2 and A3 remain `NOT_SELECTED_EVIDENCE_GAP`.

The final archive copies the exact reviewed launcher admission result and its two raw rounds and verifies their wrapper-declared SHAs again. It also records SHA-256 and byte size for `server.log`, `guard.log`, and `round16-a1-driver.log` after cleanup.

Raw completion does not select or enable A1. Device review must still validate source identity, both block receipts, canary quality, finite logprob, observer completeness, the request-level Decode-heavy gate, and the 12-attempt final ledger. A2 and A3 are outside this queue.

## 5. Offline comparator after both fresh blocks

This step remains offline, but it is valid only after each raw receipt and observer JSONL have been frozen with companion metadata containing `receipt_sha256`, `rows_path`, `rows_sha256`, `synchronize_inserted`, `query_policy`, `source_identity`, and `aggregations`. The metadata's rows path must be identical to the raw receipt's `observer_path`.

```python
from round16.a.fresh_a1_analysis import load_evidence_bundle, write_analysis_outputs

block_a = load_evidence_bundle(
    "<FRESH_A_OUTPUT>/round16-a1-raw-receipt.json",
    "<FRESH_A_OUTPUT>/round16-a1-observer-metadata.json",
)
block_b = load_evidence_bundle(
    "<FRESH_B_OUTPUT>/round16-a1-raw-receipt.json",
    "<FRESH_B_OUTPUT>/round16-a1-observer-metadata.json",
)
write_analysis_outputs("<REVIEW_OUTPUT>", block_a, block_b)
```

The comparator allows A2 only when both blocks show a primary direct Host sync exposed-wait interval of at least 10% of a directly observed Decode phase wall and the dependency proof identifies an unused same-layer independent NPU window. It allows A3 only when both blocks show a primary direct Host launch/metadata interval of at least 10% of Decode wall and that region is outside the old resident-only graph. Both ratios stay entirely in `HOST_MONOTONIC_NS`; GDN and NPU Expert kernel timings are diagnostic only. Missing gate evidence produces `DEVICE_EVIDENCE_PENDING`, while complete evidence that does not yield one consistent gate produces `NO_CANDIDATE`. Any missing request, required layer/event, event identity, source/token identity, or unsafe aggregation raises `EvidenceContractError` before a decision artifact is written. CPU Expert remains null and excluded while the native worker callback is unavailable.
