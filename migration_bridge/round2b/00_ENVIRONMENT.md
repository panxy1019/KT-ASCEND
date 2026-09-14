# Round 2B Environment

## Frozen repository

- Round 2A final SHA: `540ccbc28b1fa9327b90dc86c244e0df707409d8`
- local immutable marker: `round2a-a3-verified`
- Round 2B branch: `feature/kt-ascend-runtime-plane`
- isolated A3 audit copy: `/home/admin/kt_round2b_540ccbc`
- raw evidence: `/home/admin/kt_round2b_540ccbc/logs/round2b/`

## A3 host and disposable container

| Item | Observed value |
|---|---|
| Host | openEuler 24.03 LTS-SP1, kernel `6.6.0-72.0.0.76.oe2403sp1.aarch64` |
| CPU | Kunpeng 920 7280Z, 640 logical CPUs, 8 NUMA nodes |
| Container | `kt-r2b-ascend-plane`, Ubuntu 22.04.5, aarch64 |
| Image | `quay.io/ascend/verl:v0.8.0-cann9.0.0-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-vllm` |
| Device exposure | physical NPU 0 only, reported as `Ascend910_9382` |
| Python / compiler | Python 3.11.15, GCC 11.4.0, CMake 4.4.2 |
| CANN | 9.0.0; OPP timestamp `20260428_134817545` |
| Driver | package `26.0.rc1`, inner `V100R001C10SPC001B257` |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post2` |

The container was non-privileged, limited to CPUs 0-7, 32 GiB memory and 2048 PIDs. It received only the four device nodes required for physical NPU 0 plus read-only driver/firmware mounts. Existing cluster containers and NPU 1-7 were not modified or stopped.

Ordinary build packages (`pkg-config`, `libhwloc-dev`, `libnuma-dev`, `ninja-build`) were installed only inside this disposable container. No package manager or configuration change was made on the host.

Primary evidence: `environment.log`, `container-inspect.log`, `npu-before-container.log`, `deps-install-final.log`.
