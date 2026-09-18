# Round8 NPU namespace recovery — 2026-09-18

The first config-only success did not prove device usability. A real tensor
probe failed with zero devices and aclInit 507899 / Resource_Busy. Failure logs
are preserved under `/home/admin/kt-round8-qwen35/` on .22.

Changing ASCEND_RT_VISIBLE_DEVICES (unset/2/0), prioritizing driver libraries,
disconnecting bridge networking, and matching the existing SYS_NICE/container
network configuration did not fix enumeration. These are negative controls,
not production numerical changes.

The known-good Round7 container still enumerated one device. Host dmesg showed
`uda_occupy_dev_by_ns` / `Conflict open udevid` for physical device 2. Thus an
idle container may retain driver namespace ownership even with no NPU process
in npu-smi. Device health and empty process listing are insufficient admission.

Verified `kt-r67-c64-development-v1` contained only `/bin/sleep infinity`, then
stopped that completed project-owned container. Its files/evidence remain intact.
No unrelated container or hardware reset was used. Repeating the same tensor
probe in the new Round8 container then printed:

```text
NPU_CONTEXT_PASS Ascend910B3
```

Follow-up independent process: 20 repeated transfer/add outputs exact,
`NPU_TRANSFER_ADD_20_REPEAT_PASS Ascend910B3`.
Remote raw log SHA256:

- `npu-context-release-owner-v1.log`:
  `36baf2dbcd4d3b4fbf8f8c865cd6185e4731cd2b70b3962e6ce60f1fef52f240`
- `npu-context-regression-v1.log`:
  `340234c75d3ec0ff380d60ff37aa5a8c19f284868bff4cb39d24be3a7ca2fc3d`

The probe asserts one visible device, host→NPU transfer, accelerator add,
NPU→host transfer, synchronization and exact FP32 result. This is a primitive
PASS using the read-only reference Python environment, NOT Qwen/GDN/kernel or
standalone-build qualification. `npu-smi` executable warning remains nonfatal.

Current container ID:
`86b15b6fa8158185ac24f81cf53ff293f96c70b2a0c51e219b9545f81ed00e15`.
Current launch: network none, SYS_NICE, ASCEND_RT_VISIBLE_DEVICES=0, physical
NPU2 only, CPU128–191/NUMA4,5, 192GiB; previous mounts remain read-only.
Failed container retained as `kt-round8-qwen35-prep-driver-failed-v1` (stopped).
Do not restart the old Round7 NPU2 container concurrently with Round8.

Official container-device reference consulted:
https://www.hiascend.com/document/detail/en/mindcluster/730/clustersched/schedulingug/dlruntime_ug_004.html
This documents device mounting controls; the namespace-conflict attribution
above comes from the actual host kernel log and before/after reproduction,
not a claim that the documentation proves this specific failure.

Next: independent dependency lock, CPU512/C64 and GDN probes, real sample
conversion, then full-model static Q64→Q32. Round7 cancelled tests stay closed.
