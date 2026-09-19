# C128 CPU 扩展计划 — 2026-09-19

用户要求考虑C128。本文件只纳入独立扩展实验，不立即分配核心，不改当前C64。

## 实测拓扑

.22：aarch64，256在线CPU，SMT=1，4socket*64核，8NUMA*32核。
node0=CPU0–31，node1=32–63，依次到node7=224–255。
sysfs距离：本地10，配对node11，远端24/25/32。
宿主numactl缺失，使用sysfs读取，没有修改宿主系统。
物理资源存在不等于128核随时空闲；实际cpuset与NPU locality需在执行前准入。
当前Round8任务不迁移，不停止他人负载为C128让路。

## 候选设计

- NPU TP1、batch1保持不变，C128不是NPU TP128。
- 首选审计4个NUMA本地子池*32workers，对照C64=2*32。
- 相同权重精度、source、mask，只改变CPU执行拓扑。
- intermediate640：候选C64=320+320，C128=160+160+160+160。
- 160可整除MXFP4/Q8_0的32weight block，但不能据此宣称kernel支持；
  真实tile、prefill、down partial、量化scale分组与merge仍需验证。
- 按分片first-touch和NUMA本地放置，不复制4份全expert bank。
- 确认实际CPU runner支持多子池；不要只把workers改128或跨NUMA绑大单池。

## 收益与代价

可能利用更多本地内存通道与解量化/计算并发；也可能增加跨socket访存、
barrier、4路FP32 partial合并和调度成本。每路工作更小，Top10也不保证
128线程能摊薄开销。CPU利用率高不是扩核收益证明。
需要记录NUMA页分布/副本、CPU wait、latency与吞吐，不能只看核数。

## 有界顺序

1. C64先完成精度能力探测，选定相同格式后比较C64/C128。
2. 局部真实expert/Top10/grouped-prefill/decode、权重/分片身份。
3. compact、同路径重复、clean restart、ownership、finite、partial merge。
4. 独占同机短中协议，预冻结重复预算，交替对照，不做全格式*全拓扑矩阵。
5. C128若作为产品候选，补冻结CMMLU256，总-5pp仍对同一模型reference。
6. 记录worker数、pool数、NUMA map、CPU set、分片/merge次序和binary身份。

跨C64/C128归约顺序可不同，不要求两配置bitwise相同；每个配置自身
必须确定，局部算子对正确reference通过。C64通过不能继承为C128通过。
性能采用标准在测试前冻结；建议以可重复E2E净收益、无不可接受workload
退化和产品质量为准，不事后追加样本直到获益。无明确业务SLA时只报告
性能边界，不编造门槛。C128无收益则保留C64，不阻塞Round9主线。

推荐顺序：BF16存储reference → MXFP4/Q8_0候选 → 同精度C64/C128 →
冻结默认拓扑 → PLE/调度；格式探测不通过时不得用扩核绕过数值/系统门禁。
