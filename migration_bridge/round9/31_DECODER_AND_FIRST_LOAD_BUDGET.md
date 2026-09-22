# Five real decoder gates and monitored first-load budget

2026-09-20,110.123.0.3 physical15. Both schedules remain PAUSED.

Five independently constructed real decoder layers0/3/23/24/47 passed whole17,
chunks3+4+10 and17 single-token calls, each repeated3 times:45 runs total.
Actual router/TopK, Q64 original BF16 resident experts, LLAMAFILE C64 CPU experts,
shared expert and GR are connected. GDN layers0/24 had zero output/state path
error; QSA layers3/23/47 maximum output relativeL2 was0.00100107846<0.01.
Highest NPU allocation was1,232,061,440bytes. These are single-layer gates,
not48-layer, embedding/logits, scheduler, quality or performance qualification.

Two production seams were repaired: checkpoint CPU tensors now use owned
per-tensor staging before NPU copy, and decoder construction directly registers
KTEP before parent MoE creation. All five tests started with KTEP absent from
sys.modules and verified the constructor created the actual wrapper/runner.
The loader-map probe still checks1658 source tensors/1094 destinations and128
PLE parts without allocating full model weights. Nine isolated contracts pass.

## Training and memory accounting

The user answered15G to the question asking the verified GiB peak during this
round, with the currently observed8.8GiB physical-board reading as context.
Record15GiB=15360MiB as USER_ATTESTED_WORKLOAD_ENVELOPE, not an assistant-
measured peak or enforced reservation. Accounting uses npu-smi physical board
capacity65536MiB and **nonowned board occupancy**, including system baseline.
PyTorch visible capacity65796046848bytes is a different domain; the8GiB reserve
is physical-board reserve, not8GiB additionally beyond unavailable system HBM.
The guard separately records process totals and board totals and rejects a
nonowned board reading above15GiB. New training PIDs require renewed inspection.

Exact payload: routed weights28800MiB, other weights9436.989MiB, known states
319.679MiB rounded to321MiB. Static trial allowances1536MiB each for runtime
workspace and load temporary yield41630MiB, within41984MiB owned budget.
The workspace allowances are NOT measured full-model peaks. A first run may
fail or be stopped. A successful load alone cannot validate lazy workspaces.

Small-allocation probe verified the process-local PyTorch cap API rejects a
4MiB allocation under1MiB cap. It does not constrain native/CANN allocations.
The real worker uses39.5GiB PyTorch cap, leaving1.5GiB inside the41GiB owned
budget for unmanaged allocations. A host monitor is armed before docker start,
samples every1s plus command duration, and stops only the owned container above
40.5GiB owned or55.5GiB whole-board HBM, before the41/56GiB budget boundaries.
It also stops on nonowned usage>15GiB, new unbound device process, unavailable
telemetry, host available RAM<64GiB, or the bounded monitoring window ending.
This is a reactive guard, not an atomic hardware reservation. Its early
threshold is intentionally below the conservative summed static allowances.

CPU visibility is whole NUMA6/7 (480–639,160 logical/80 physical cores), while
quota72 supports64 workers plus8aux budget. The admission checker must represent
these separately. Container RAM/swap cap640GiB, host reserve64GiB, disk floor
1.2TiB, no external network and only physical15 are retained.

Only STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN may be granted now.
FULL_MODEL_PEAK_NOT_MEASURED remains until actual loading plus bounded first
prefill/decode and the4096-token workload boundary are observed. Never relabel
this static review or single-layer PASS as full-model resource/quality PASS.
