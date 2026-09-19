# .16 secondary-host audit — 2026-09-19

Result: INFRASTRUCTURE_BLOCKED_NPU_NAMESPACE. Stability and Dynamic are NOT_RUN
successfully, not qualified. Prior controller dispatch did not imply readiness.

## Verified metadata

- Host gpu-server-02, Ascend910B3; driver25.5.1/ascendhal7.35.23 matches .22.
- Frozen image b81e664642aecd5f4cc8ec38b233aae38f65b9cf7feea40e54e13c08cd319347.
- Model revision59d61f3ce65a6d9863b86d2e96597125219dc754.
- CPU manifest35ca0ac63a3646ba7b89675053bbead19724325fa9d73f99910ef3c156a84405.
- Q32 mask b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486.
- Command b49d2a0b03ab8baa5be2d4e6da5f65a74b1e9505f6c77b4dccf91c123bb39053.
- Container CPU128-191, NUMA4/5, memory/swap limit384GiB, physical NPU2.
- Transfer completed. These are metadata checks, not a fresh hash of every
  transferred checkpoint/GGUF byte. Full content audit remains pending.

## Failure and controls

Server-A failed during import with `torch_npu detected, but NPU device is not
available or visible`. Independent device_count returned0. Unsetting runtime
visibility did not fix it. Adding SYS_NICE to match .22 did not fix it.
Host dmesg repeatedly reports `uda_occupy_dev_by_ns: Conflict open udevid`,
udevid2. `/proc/uda/namespace_node` shows ns_id0 rooted at PID370374 covering
devices0-7. Docker inspection maps PID370374 to `mzy-mindspeed`. Other existing
containers also mount all devices. None were changed.

Official guidance identifies ns_id<128 ownership and root_tgid as the
diagnostic path for this failure:
https://www.hiascend.com/document/caselibrary/detail/topic_0000002238354198

The failed original project container was stopped and renamed
kt-round8-qwen35-stability-failed-device-v1 to preserve its filesystem. Raw logs
remain in /home/admin/kt-round8-qwen35/qwen-q32-stability-v1 on .16. A capability
control replacement also failed visibility and was stopped. No model/operator
change is justified by this failure.

## Next gate

Obtain an available namespace/device allocation without modifying others'
workloads, then require in-container enumeration, device allocation and tensor
copy/synchronize before model startup. Use a new stability campaign identity.
Preserve existing failure files. Complete content hashes and actual software
version audit before promotion. .22 performance remains a separate campaign.
