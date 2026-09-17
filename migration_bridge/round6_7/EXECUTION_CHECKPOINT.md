# Round6.7 execution checkpoint

## Frozen and launched (not yet qualified)

Current study: `/workspace/results/r65-r67-serial-v3` in the existing container.
Study freeze SHA256 `11ada1752b5e5c0571920ab6536421d3193e73afa6cc2f630e076d469fef693c`.
CMMLU256 manifest SHA256 `bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04`.
Compact manifest SHA256 `555f578af6d816b25ee5efe6590a472b654f27a74614c011bba599a8f8c24933`.

Profile JSON hashes (exact preexisting all-layer mask bytes):

| Profile | SHA256 |
|---|---|
|R64|`5d1cbbbfcaaa88582e60ff2ffeeaa0341a525cf5e2a2dac7400688732846a158`|
|R32|`4fbcf329e50426a5fcfa9cfcf98277ad7ed69910793eef139ee27560b3b394b3`|
|R16|`27433a83062c6eda0fe9b957c772ff96c01e5069fabe970ffa892d595efe3e0c`|
|R8|`f2ae533c52318cb5015ba6396c2329865abdb157c2e774061eb01557c4d40c0d`|

Controller: `run_serial_sweep.py`, wrapped by retained owned-process guard,
48h maximum overall / 12h per model session / 1h per quality request.
Only one server at a time. Auto advances system→CMMLU→next profile, then
accepted-profile performance. A per-profile system failure excludes its quality
and performance; R64/provenance failure stops the study. No promotion merely
from launch. No automatic new container, Dynamic or Deferred development.

Outputs: controller.log, progress.json, per-profile lifecycle/system/quality/
performance artifacts, host-telemetry.jsonl, final summary.json and artifact
hashes. Final Pareto selection requires reviewing measured capacity/performance.

## Preserved initial infrastructure failure

Study v1 freeze `c61c9c1c380e8863a004c6b1d7f2bd780da6333f6afe4d5252683f42706e696a`.
R64 never loaded: ATB set_env.sh line43 read undefined ZSH_VERSION under
nounset. Original guard/process.log and failure summary retained in v1.
v2 launcher follows the retained launcher's set+u/source/set-u convention,
correct CANN IR environment, no-login shell, and correct health-generation OFF
flag. This is test-tool infrastructure correction, not production or threshold
change. v1 and v2 corpus/profile bytes are identical.

v2 freeze `d1415f886eeb6a2d33fdce564a75db60707c7227c754dbb882dba837e9ab2cd3`
then failed before readiness after model load: the test launcher overwrote CANN
PYTHONPATH, so TBE import failed during GE initialization. v3 appends project
paths while preserving CANN paths, exactly as the retained launcher does.
No inference-quality results existed in either failed attempt; no threshold,
input, expert mask or production implementation changes were made. Guard logs,
exception/lifecycle records and distinct frozen snapshots are retained.

v3 also distinguishes per-layer-token vs whole-model-token routing denominators
and refuses acceptance after any later performance-stage execution exception.
Host HBM parser now handles npu-smi25.5.1 `910B3` name and shared table columns;
host sampler SHA256 `0b7f4152d59f6a9f7b35461a53ce7b377dc9bd689a2c208a8098ace5c6d4d22a`.
R64/Hybrid run infrastructure and generation protocol remain otherwise frozen.

LOCAL_VERIFIED: 12 contract tests, Python compilation, bash syntax, diff checks.
Full target qualification remains pending until execution artifacts close gates.
