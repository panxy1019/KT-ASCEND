# Round9 candidate framework delivery — 2026-09-20

Scope: the development-only acceptance criteria in35, explicitly authorized by
user to execute continuously. No NPU, full-model, heavy bank conversion, or
performance task was run in this batch. Both schedules remain PAUSED.

## Delivered capabilities and honest boundaries

| Area | Implemented and verified | Remaining before candidate full-model use |
|---|---|---|
| Candidate configuration | Separate schema, C64/C128/storage/placement validation, topology capacity plan, exact runtime-config guard; no silent fallback | Fresh CPU/NPU/training/memory/source admission; no plan grants reservation |
| C128 | Four32-worker args, four160 intermediate slices, physical-core/SMT/auxiliary checks; existing C++ partition helper compiled on x86 and ARM; opt-in isolated SGLang model seam | Actual128-worker native expert execution and NPU system tests; no full native-extension rebuild or C128 qualification claimed |
| Low-bit | BF16 packed checkpoint reader; MXFP4/Q8_0 block encoder/decoder; streaming GGUF writer, independent library read/dequantization; per-projection source hashes; disk preflight; no-clobber publication | Actual320/160 ARM grouped-prefill remains UNSUPPORTED_NATIVE_KERNEL_160_320_GROUPED_PREFILL; true native tail implementation/qualification must precede full low-bit model tests |
| Dynamic | Lease/generation-bound observations, deterministic slot-preserving selection, atomic logical publication under lock, concrete original-BF16 Torch resident copy and in-place KTEP mask/map updates | Generic SGLang request cleanup/scheduler hook is not installed. Its candidate config rejects with DYNAMIC_SCHEDULER_BOUNDARY_INSTALLATION_AND_DEVICE_GATE_REQUIRED. Adapter can be exercised by a separately guarded probe; it is not production Dynamic qualification |
| Workflow | Independent profile/plan/check commands, source hashes and evidence; isolated overlay installer copies baseline and binds complete payload tree | No automatic device launcher or performance stage |

This meets the stated framework criterion: execute real host/tensor seams or
reject with a specific capability gap. It does not claim all three methods can
already be switched on for a full-model run. In particular, tomorrow must not
skip the low-bit native implementation gap or the Dynamic scheduler boundary gap.

## Validation and evidence

- Local Python3.13/x86:23 host/configuration/codec/state/installer/workflow tests PASS,
  including compilation and execution of the actual C++ partition header.
- A3 Python3.11/ARM container image
  `sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a`:
  all27 tests PASS, including real CPU Torch tensor promotion/mapping, in-place
  mask pointer preservation, independent GGUF reader/dequantizer, and nonconstant
  blocks at multiple magnitudes. Expert fixtures have different logical-expert
  and gate/up/down values, so an ID/packing mistake cannot hide behind equal data.
- All test containers:2CPU quota,4GiB memory and equal memory+swap cap,
  networknone, privilegedfalse, Devices=[]; all exited. Containersv1/v4 are
  preserved expected failing-test stages (missing adapter/factory), not NPU failures.
- Independent review used the existing gpt-5.6-sol/medium agent. It found identity,
  lease, plan-completeness and installer-isolation issues; all were fixed and
  rechecked. No unresolved host/framework blocker was reported.
- `evidence/development-20260920/local-final/` and
  `evidence/development-20260920/arm-cpu/all-final/` contain logs and source hashes.
  `arm-cpu/containers.json` records actual limits/devices/terminal states.

Baseline profile SHA remains
`2a88116fb6a67f31149494104106857a88ba4fcf4078a6b2606b3d1fe64e69c4`;
baseline model overlay SHA remains
`f333b905c6534f65a41795ba3a99605fc1f67a4e4f8659991436c7f706f9c361`.
Remote v6 baseline source/native runtime were not modified. The CPU-only candidate
workspace is `/mnt/data1/kt-round9/candidate-dev-20260920`.

## Entry points

From the repository root, use a NEW output path per run:

```bash
python3 migration_bridge/round9/tools/candidate_workflow.py check \
  --stage host --output /tmp/r9-host-review
```

Stages are `host`, `native-partition`, `tensor`, `all`. `host` includes the small
partition compiler check and needs g++; `tensor` additionally needs torch,
numpy and gguf. It disables Torch device backend autoload. The successful ARM
run used the CPU-only container above, not device mappings.

Candidate configuration uses explicit NUMA selection after inventory; the variable
below must contain approved NUMA IDs, not inferred from old .22 topology:

```bash
python3 migration_bridge/round9/tools/candidate_workflow.py profile \
  --workers 128 --storage BF16 --placement static \
  --numa-nodes "$R9_APPROVED_NUMA_NODES" --output /tmp/r9-c128.json
python3 migration_bridge/round9/tools/candidate_workflow.py plan \
  --profile /tmp/r9-c128.json --topology /tmp/r9-allowed-cpus.json \
  --output /tmp/r9-c128-plan.json
```

Topology input is `{"quota":136,"rows":[{"cpu":480,"core":"socket:core","numa":6},...]}`.
Rows must represent the fresh allowed host/cgroup CPU intersection with true
physical-core identities. SMT siblings share the same `core` string. The output
is HOST_PLAN_ONLY_DEVICE_ADMISSION_PENDING even for valid rows. There is no
resource allocation, pinning or capacity guarantee in this command.

An independent overlay can be installed into a NEW sibling directory:

```bash
python3 migration_bridge/round9/tools/install_candidate_overlay.py \
  --baseline-python /mnt/data1/kt-round9/src/sglang-round9-port/python \
  --output /mnt/data1/kt-round9/candidate-python
```

Set `ROUND9_CANDIDATE_PROFILE` only when using that copied Python tree and newly
admitted candidate runtime arguments. Default behavior retains the frozen baseline
guard. The old full_model_server.py intentionally still requires its frozen
baseline hash; do not bypass it to launch candidates. Overlay receipt identifies
copied code, not inherited device qualification.

The converter must run with the verified original model read-only at one of:
`/models/de4b8e4d43b917e7706784d8bb445c9af86a3540` or the approved host model root.
An arbitrary same-size copy is not an accepted source. Example inside a CPU-only
container with `/models` read-only and output writable:

```bash
python3 migration_bridge/round9/tools/convert_candidate_bank.py \
  --model /models/de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --receipt /support/destination-verification/verification-1789886190714568317.json \
  --layer 0 --experts 0,1,510,511 --storage MXFP4 --output /out/mxfp4-layer0-sample
```

Replace MXFP4 with Q8_0 for the fallback. Default maximum sample4 experts;
`--full-layer` explicitly selects all512, but do not run it before native format
qualification. Partial expert banks carry `logical_ids` and
`complete_cpu_backing=false`; they cannot be a production512-expert backing.
Completion means PASS_CONVERSION_ONLY_NOT_NATIVE_QUALIFIED. Failed partial files
are retained without a completion receipt. No overwrite or all48 loop is implicit.

## Dynamic integration contract

`TorchKTEPAdapter(layers, methods, source, complete_cpu_backing_ids=..., host_staging_limit=...)`
binds real processed transposedBF16 resident tensors and KTEP methods.
`bind_controller(adapter)` validates actual maps/masks and derives resident slots.
The caller must verify full CPU backing/source identities before constructing it;
declared IDs alone do not independently verify bank contents. The original source
reader validates structure and change detection, not a substitute full model hash.

1. `generation = controller.begin(request_id)` before inference entry.
2. `controller.observe(request_id, generation, layer_id, logical_ids)` from routes.
3. Finish real KV/GDN/PLE cleanup and device consumers, then call
   `controller.finish(request_id, consumers_drained=True)`.
4. `controller.commit(controller.plan())` at the drained boundary.

All routes require current lease/generation. Planning while active is rejected.
New observations invalidate old plans. Commit must include the exact complete
current plan; a no-op closes the statistics window. Surviving logical experts
keep their slots; incoming IDs are deterministic. Copy precedes maps/masks and
all publication is under the request lock. Any copy/sync/publication exception
poisons the controller as BROKEN_RELOAD_REQUIRED. There is deliberately no fake
metadata-only rollback or reset method.

Each original expert is9.375MiB at production geometry; the host adapter requires
at least4 payloads of staging allowance (37.5MiB). It reads/copies only changed
experts, never creates a full GPU expert layer. It uses the original BF16 source
for promotion, never a decoded low-bit CPU bank. NPU adapter use requires explicit
`allow_unqualified_npu_probe=True` only inside separately admitted test harnesses;
CPU tests do not establish NPU NZ copy/workspace correctness.

## Codec provenance and boundaries

`R9_GGML_BLOCK_CODEC_V1` fixes32-weight blocks, GGUF MXFP4 type39/17 bytes and
Q8_0 type8/34 bytes. MXFP4 uses half-block nibble order, power-of-two amax exponent
and first-index ties; underflow exponent clamps to0. Q8_0 uses F32 amax/127,
round-away-from-zero and FP16 stored scale; nonfinite or unrepresentable scale
is rejected. These are converter rules, not a model-quality guarantee.

The layout/reference policy was checked against the local decoder and the
[GGML reference quantizers](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c),
then independently decoded with the actual container's GGUF library. Consulting
that source does not change the frozen runtime pin or qualify an ARM kernel.

## Remaining execution order

Keep full256 baseline incomplete until the preserved second128 is completed and
raw-audited. Tomorrow first refresh resource admission. C128 native expert probes,
low-bit native-tail work and same-quantized-weight probes, and a guarded Dynamic
adapter probe are distinct next gates. Full Dynamic HTTP testing also needs the
real scheduler cleanup seam. Combine only qualified paths; no performance work
has been scheduled. No push, main merge or tag.
