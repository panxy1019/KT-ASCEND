# First full-model load and Ascend pool dispatch repair

2026-09-20, physical15 on110.123.0.3, shared training left untouched.

Trialv1 was stopped before docker start because this host has no kernel PSI
memory file. PSI is now an explicitly unavailable optional metric. Mandatory
board HBM, process identity and host MemAvailable checks remain fail closed.

Trialv2 completed reading all131 original checkpoint shards and initialized
all48 native C64 expert backends. The actual SGLang loader reported weight load
elapsed146.25s, memory37.92GiB. The HTTP server became ready at14:32:40 UTC.
Observed owned process HBM peaked39932MiB; boardHBM peaked48851MiB across this
run including teardown. This is evidence of actual full loading, not a passing
full-model forward, peak-workload qualification, or task-quality result.

The first HTTP request (Hello,4 generated tokens requested) failed at layer3:
QSA pool APIs absent. Root cause: init_memory_pool's generic Ascend branch ran
before the later hybrid branch, allocating NPUMHATokenToKVPool for all48 layers
instead of Round9QSAKVPool's12 full-attention layers plus QSA side states.
Single-layer fixtures had constructed the correct pool directly and therefore
did not cover this dispatch seam.

Patch0004 excludes only qwen4_exp_text from the generic Ascend branch, allowing
it to reach the existing Round9 hybrid pool constructor. Other Ascend models
retain their original dispatch. A new small NPU probe calls the real
ModelRunnerKVCacheMixin.init_memory_pool, with32token allocation, actual48layer
IDs and small synthetic recurrent state shapes. TP size is supplied as1;
this does not test distributed startup. Positive and ordinary-Ascend negative
controls verify actual pool types and writable PLE/GDN/QSA states.

Probe bringup initially found fixture omissions: uninitialized TP group and
utility expecting device='npu' rather than'npu:0'. These failures are preserved;
neither is a production repair or a false QSA dispatch RED.

After the actual scheduler exception, NPU process rows disappeared before board
used HBM dropped. The monitor conservatively subtracted owned=0 and stopped
its own container. Docker OOMKilled=false; exit137 followed owned guard cleanup.
Do not describe that transient as an observed training spike or capacity OOM.
The raw monitor and server logs preserve ordering. The next monitor version
also handles this host's hybrid cgroup layout: use actual memory.current when
available, otherwise the memory controller's v1 files. It retains fail-closed
HBM cleanup behavior.

Next run enables an optional observational log of per-forward torch allocated,
reserved and their peaks, alongside the independent host process/board monitor.
No tensor values, execution branches or tolerances are changed by this observer.
Full-model trial must be repeated after dispatch regression passes.

Real dispatch RED-v3 reproduced the exact wrong pool assertion. GREEN-v1
passed both real-constructor controls, including12QSA full layers versus48
ordinary Ascend layers and live PLE/GDN/QSA writes. Container exited0.
Trialv3 starts with this source and refreshed independent monitor/admission.
