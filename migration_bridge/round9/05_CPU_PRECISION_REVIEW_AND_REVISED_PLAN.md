# Round9 CPU 精度复核与修订建议

日期：2026-09-19。性质：按用户要求重新制定的精度路线；不是新内核已通过。
不改变正在运行的 Round8，也不更换下载中的冻结 BF16 source。
官方 KT 审计快照：f7607c0f66c643220ff43b822b24483c98c41392。

## 1. 结论

不应把“全量450GiB F32 expert bank”设为不可绕过的部署前置。
建议分开三个角色：

- 数学/局部调试 reference：按需展开少量真实 experts 到 F32。
- 首个无新增权重量化的全模型 reference：优先验证 BF16 权重存储，
  F32 输入/计算工作区/累加的 LLAMAFILE ARM 路径。
- 容量与性能优化主候选：MXFP4；唯一优先质量/工程 fallback：Q8_0。

完整 F32 路径保留为有证据需要时的 fallback，不立即生成450GiB文件。
BF16 存储不等于 BF16 累加，也不等于 Kunpeng 有原生 BF16 指令。
NPU仍BF16，CPU/NPU partial的既定FP32合并与最终cast契约不随权重压缩改变。

## 2. 官方实际采用什么

| 官方材料 | 实际格式/平台 | 可借鉴边界 |
|---|---|---|
| Ascend DeepSeek-V4 Flash教程 | CPU MXFP4 + LLAMAFILE，NPU W8A8；列出Kunpeng920/910B3 | 这是ARM低比特的直接先例，不是Qwen3.8 shape/质量证明 |
| Kimi-K2 Thinking native教程 | CPU/GPU RAWINT4 | 原生量化模型及x86/NVIDIA，不可直接把RAWINT4类搬到Kunpeng |
| KT native precision教程 | BF16、FP8、FP8_PERCHANNEL、RAWINT4 | 列出的AVX512/AMX要求不能继承为ARM支持 |
| KT-Kernel README | AMX INT4/INT8；LLAMAFILE GGUF Q4/Q5等 | 格式/ISA/模型绑定；README还提醒某些模型INT4质量可能明显下降 |

不能用这些选定案例统计所有官方部署的市场占比。能确认的是官方容量/
性能路线大量采用4/8bit，而不是要求FP32权重池。

官方 moe.hpp 的 `kt_effective_vec_dot_type()` 明确处理 aarch64无SVE：
BF16 weight的输入有效类型设为F32，走 Atype=BF16/Btype=F32/Ctype=F32。
本地同函数也存在。这里只是CODE_INSPECTED；实际wheel、640shape、
C64子池和数值边界仍必须重新测试，不能仅凭函数存在写READY。

## 3. F32的必要性应重新解释

若source权重本来是BF16，精确展开成F32不会恢复不存在的权重有效位；
它扩大存储并可能改变kernel/运算路线。以往F32是控制变量、复用已验证
CPU路径的工程选择，不是模型要求每个权重都以32bit驻留。
FP32累加与FP32权重存储是两个选择，低比特权重也可保留FP32累加。
降低存储可能减少DRAM读取，但反量化/打包/向量实现有成本，不能许诺
8倍容量下降就8倍加速。

FP16不是原BF16的无损替代：指数范围不同，不能仅因都是16bit就换。
下载的完整BF16 source仍有必要：它服务身份、NPU resident/promotion和
独立量化来源，而不是意味着还要强制再存一份全量F32。

## 4. 容量预算（全48*512 backing，不仅当前CPU-owned）

参数量48*512*3*2560*640=120795955200。
以下为矩阵block payload，不含文件metadata、padding、运行时副本/缓存。

| 权重格式 | 每权重平均存储 | 全expert pool GiB |
|---|---:|---:|
| F32 |32bit|450|
| BF16 |16bit|225|
| Q8_0 |34bytes/32weights=8.5bit|119.53125|
| MXFP4 |17bytes/32weights=4.25bit|59.765625|
| Q4_0（非本轮主候选） |18bytes/32weights=4.5bit|63.28125|

block比例来自GGML公开布局；必须对实际KT子模块版本再验，不能当作已
生成文件大小。PLE仍按原精度，约95GiB的估计不随CPU专家量化消失。
Q64 NPU routed payload仍28.125GiB；CPU lowbit不自动减少这项HBM。
保留原source的mmap页、加载匿名副本、C64双子池分片和PLE会增加实际峰值。

## 5. 一个重要的shape约束

hidden2560，intermediate640，候选C64分片320+320。
Q8_0/MXFP4的32weight block：2560/640/320均可整除32。
常见Q4_K的256weight superblock：640和320不能整除256。
因此不能仅因旧模型用Q4_K_M就照搬。down reduction轴、切分方向、
padding/裁剪与各kernel tile限制要逐项审计。可整除32也不是内核可用证明。
Q4_K_M还是混合tensor格式策略，不是所有tensor恰好4bit。

## 6. 修订后的最小执行路线

### P0：先局部格式能力探测，禁止先做全池转换

保持冻结source、真实logical expert和真实shape。在独立Round9环境做
早/中/末层、GDN/QSA后MoE、decode/grouped-prefill和C64分片测试。

1. BF16 storage LLAMAFILE vs 同值F32 canonical及现有F32 production。
2. MXFP4转换/回读/packed block与ARM执行；量化kernel对同一量化权重
   解码reference，不把量化损失当作kernel错误。
3. Q8_0作为唯一优先fallback，检查激活vec-dot类型与累加/缩放边界。

分别记录weight dtype、输入/临时激活类型、GEMM累加、SwiGLU、route
weight与partial merge类型。不可把权重4bit宣传为整个模型4bit。
没有实际支持前不声称Kunpeng具备原生FP4/BF16 ISA。

### P1：无新增权重量化的工程reference

若BF16存储通过局部安全、真实C64边界与系统门禁，优先构建Q64/BF16
CPU storage reference，不再全量F32展开。保留有限F32算子reference。
这会改变旧计划的reference实现名称；必须在首个模型质量结果前冻结
新协议和ID，例如QWEN38_Q64_BF16_STORAGE_REFERENCE_V1，记录相对
旧F32计划的变更原因。不能仍标作已执行Q64/F32。

BF16路径若失败，定位具体实现而不是假设精度不足；只有确有需要时
回到F32 reference并重新资源准入。未通过不能用量化模型互相接近代替。

### P2：MXFP4产品候选，Q8_0退路

官方DSV4的MXFP4来源不等于对本BF16 Qwen checkpoint做post-training
MXFP4转换必然保持质量。先冻结converter规则/scale/rounding/packing，
从同一BF16 revision转换，不下载来历不明的混合量化权重替代。

MXFP4需要同量化权重reference局部门禁、系统compact/确定性/重启、
冻结CMMLU256相对同一个无新增量化reference的observed delta>=-5pp。
若实现不合适或质量失败，记录原因，走Q8_0；不提高门限/换题/重新挑mask。
若都失败，保留BF16静态路径，lowbit子目标如实未完成。

### P3：精度冻结后再叠加PLE与调度

选择依据：系统正确 → 质量护栏 → 实际容量 → 同协议性能。
速度无收益但容量有收益可以限定采用；不得承诺MXFP4一定比Q8快。
后续PLE异步、Dynamic、Q32均对同一个原reference核算总质量差。
promotion仍读原BF16source，不能把有损CPUbank反量化冒充原权重。
转换覆盖全部512experts/层，为demotion保留完整CPUbacking。

## 7. 当前动作边界

本次只形成精度复核与新计划，未切换任何运行模型、未量化、未修改
Round8 gate、未生成全池F32。下载照常，正式Round8性能时暂停。
在正式转换前完成P0与reference协议冻结；尚无任何新格式A3 PASS。

## 官方来源（2026-09-19核查）

- https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md
- https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/doc/en/kt-kernel/Kimi-K2-Thinking-Native.md
- https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/doc/en/kt-kernel/Native-Precision-Tutorial.md
- https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/kt-kernel/README.md
- https://github.com/kvcache-ai/ktransformers/blob/f7607c0f66c643220ff43b822b24483c98c41392/kt-kernel/operators/llamafile/moe.hpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
  （格式定义参照；不是当前部署子模块pin）
