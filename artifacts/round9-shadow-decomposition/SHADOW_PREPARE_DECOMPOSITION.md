# Expert Shadow 准备成本分解：本轮执行状态

日期2026-09-22。状态 **BLOCKED_WITH_EVIDENCE / CPU_ADMISSION_DENIED**。

本轮已完成计划修订、真实接口审计、A/B fixture定义、默认OFF观测实现、独立审阅修复及离线测试。正式OFF启动前，既有准入函数拒绝所选CPU负载；未创建或启动物理组件容器。没有本轮设备时间、数值、HBM峰值或OFF/ON实测结果，以下不填造零值，也不继承v20结果当本轮通过。

## 可用的实现

真实prepare仍由原 `TorchKTEPAdapter.copy_experts` 执行。仅在新的独立snapshot中添加上下文观测，原v63/v20未改动。

| Phase | 真实操作 | 口径 |
|---|---|---|
| source_file_read | 原reader的open/seek/read | Host调用区间和logical bytes；不是磁盘带宽 |
| host_materialization | bytearray/frombuffer/reshape/finite | 包含嵌套source_file_read，不与之相加 |
| host_projection_build | gate/up拼接、转置或down读取 | 包含materialization；维持原步骤 |
| nd_allocation | 原临时staged创建 | 记录shape/stride/storage pointer/bytes及layout；没有新tensor |
| copy_call | 原blocking copy_ | Host区间和独立device event；不称纯enqueue或DMA |
| nz_index_copy | 原NZ parent index_copy_ | 独立观测，不重写布局算法 |
| existing_sync | 原projection后的同步 | 次数和顺序不变，不增加每阶段全局同步 |
| staging_retire | 原del staged | 位于prepare内；不等于allocator返还驱动HBM |
| consumer_drain / lease_release | 原backend synchronize/release | prepare外单列 |

Host parent覆盖为直接child区间并集；嵌套phase时长另列且不可相加。父子越界、坏区间、空trace、异常退出或观察器失败会降级。device事件只在已有drain后query/读时长，无额外同步。每事件绑定run、case、request/placement/allocation epoch和shadow lifetime。最大2048events/8MiB；metadata只读，不在热路径hash大tensor。时钟分辨率见远端cleanup记录；事件观察开销尚未物理校准，不能据残余给出未测的开销百分比。

## 数值与生命周期协议

A采用既有真实decode输入及top10路由驱动原BF16 shadow provider。B仅将最后路由230替换231，其他输入/权重不变，标记DERIVED_COMPONENT_CASE。两个集合经排序只更换slot7对应expert；当前provider仍完整复制每组10expert，未实现changed-slot reuse。

构造一次template与32-slot shadow，依次A、B，每次原execute_group→synchronize→release。CPU BF16 reference在A/B测量之后分别计算，沿用既有relativeL2<=0.01；不能用旧mixed MXFP4/BF16 output作新BF16 oracle。ON还必须与OFF输出hash和copy计数精确匹配。初始化加载template专家0–31并做身份检查，会影响缓存；OFF/ON只做组件语义及归因，不作速度提升/开销百分比。

## 资源阻塞证据

- 原选定160个physical CPUs，NUMA2/3/6/7，CPU quota136。
- `cpu-admission-detail.json` 的1秒采样平均idle约0.4057625；门槛严格>0.8。
- 初始只读检查与正式OFF launcher均报告`selected CPUs busy`。
- 卡15采样时HBM2878MiB且没有NPU进程；CPU准入独立于NPU空闲。
- 另有`kt-r7-a3-return-v1`，其显式cpuset在NUMA0/1，不能仅凭容器存在将本轮所选CPU忙碌归因于它。未修改它或训练容器。
- 未降低门槛、未换CPU集合、未做DONTNEED/drop_caches、未恢复定时任务。

固定组件限额：4% allocator（不超过2,631,841,873bytes），owned4096MiB，Host64GiB、memory-swap64GiB；原private bank300MiB、max projection200MiB、workspace512MiB合计reservation1,061,158,912bytes。基线后的执行峰值上限为projection+workspace，不能漏掉projection。共享整卡合同仍保留15GiB训练预算和8704MiB余量。

## 验证

63项本地相关测试通过；另一个最多1CPU、8GiB、120s、无NPU设备的冻结镜像检查通过78项，退出0。两组测试包含重叠计账、错误降级、provider异常保留、身份锁定、资源计算、derived fixture、CPU tensor adapter及既有Layerwise绑定。移除观测import/with/metadata语句后，两个核心文件的AST与a5b26444e基线完全一致；这不代替真实NPU OFF/ON门禁。

## 接续点

源snapshot：`/mnt/data1/kt-round9/shadow-decomposition-20260922-v1/source`；manifest在父目录`source-sha256.json`。预算未消费：OFF/ON各一次，每次1200s，自动重试0。

恢复时先重新准入相同资源和训练完整ID，再由冻结launcher执行OFF，成功后才可ON。需要先确认v21容器名仍未占用；如果环境变化，记录新身份/名称但不可静默覆盖snapshot。ON强制沿用OFF source/capture/sidecar hashes；若源码需要修复，需建立新snapshot并明确该对比较不能混用。当前尚未进入M2优化。
