# Isolated build staging

Local stdlib venv: `round8_prep/07_BUILD/venv`, created successfully; wheel/build output reserved under this directory and gitignored. This is not an Ascend runtime image or dependency lock.

Reserved future container name: `kt-round8-qwen35-prep`; ports18400(Q64),18401(Q32),18402(optionalQ16). Names/ports are proposals, not claims of availability. Before creation inspect all container names/listeners/devices. Use separate writable checkout, extension build/cache, wheel dir, virtualenv and logs. Model mount may be shared read-only only after exact identity verification.

Do not reuse Round7 cpuset128–191/NPU2 while owned run is active. C64 must retain its physical-core/subpool definition; do not replace with arbitrary64 logical threads. Discover actual NUMA/NPU locality; no fixed NPU assignment at prep time.

Build sequence after admission: freeze Round7 final SHA/runtime hashes → choose minimal Qwen dependency adapter → independent image pinned by digest → resolve torch/torch_npu/CANN/GDN ABI → import/config/tokenizer tests → build native ARM extension → same-operand probe. Save compiler flags, wheel hashes, pip freeze and actual imported paths. No host-library mutation.

Resource guard takes fresh JSON inventory and defaults to rejection when fields absent; it does not collect live data or launch anything. Retain live inventory with UTC timestamp and process ownership before using a passing result. During Round7 formal performance it rejects heavy prep even if another NPU looks idle.
