# E001 — A3 CPU ISA

状态：`PASS / A3_VERIFIED`

## Commands

```bash
uname -a
uname -m
lscpu
cat /proc/cpuinfo
```

通过 SSH 在 A3 宿主执行，只读，无容器/NPU 依赖。

## Topology

| Field | Observed |
|---|---|
| Architecture | aarch64 |
| Model | Kunpeng 920 7280Z |
| Sockets | 4 |
| Cores/socket | 80 |
| Threads/core | 2 |
| Logical CPUs | 640 |
| Physical cores | 320 |
| NUMA nodes | 8 |
| CPU lists | node0 0-79；node1 80-159；…；node7 560-639 |

`/proc/cpuinfo` 的实测 flags：

```text
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid
asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve
asimdfhm dit uscat ilrcpc flagm ssbs sb dcpodp flagm2 frint svei8mm
svef32mm svef64mm svebf16 i8mm bf16 dgh rng ecv
```

## Requirement matrix

当前 CMake `kt-kernel/CMakeLists.txt:248` 使用 `-march=armv8.2-a+fp16+dotprod+sve+bf16`。

| Feature | CMake requires? | A3 present? | Evidence | Risk |
|---|---:|---:|---|---|
| ASIMD/NEON | architecture baseline/implementation likely | yes | `asimd` | low for presence only |
| FP16 | yes | yes | `fphp`, `asimdhp` | low for presence only |
| dotprod | yes | yes | `asimddp` | low for presence only |
| SVE | yes | yes | `sve` | vector length/kernel behavior not tested |
| BF16 | yes | yes | `bf16`, `svebf16` | numerical/kernel behavior not tested |
| i8mm | no in current `-march` string, relevant to INT8 | yes | `i8mm`, `svei8mm` | kernel sources missing, usage unknown |

## Conclusion

A3 ISA 满足当前 CMake 明示的 ARM 编译 flags，且额外具备 i8mm/SVE i8mm。KML build 的首个失败不是 ISA 检测。由于未生成和运行 KML 二进制，本实验不证明 instruction correctness、SVE vector-length assumptions 或性能。

