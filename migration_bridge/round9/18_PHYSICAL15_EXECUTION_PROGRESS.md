# Round9 物理15首批执行结果 — 2026-09-20

用户已审阅并批准 `docs/superpowers/plans/2026-09-20-round9-physical15.md`。
本批结论：**CPU 有界探针通过；NPU 访问准入阻塞；全模型 NOT_RUN。**
不是 Round9 全轮完成，也不是 BF16 全模型参考已经建立。

## 1. 已执行与证据

证据目录：`evidence/physical15/20260920-admission/`，入口 `summary.json`。
服务器工作目录：`110.123.0.3:/mnt/data1/kt-round9`。

| 项目 | 实际结果 |
|---|---|
| 原模型身份 | 目的端既有完整回执 SHA256 匹配；source revision不变，未重复读取360GB做全量hash |
| live safetensors headers | 131 shards、1658 tensors、359999963128 payload bytes，结构清点 PASS |
| 主 routed experts | 96 packed tensors，241591910400 bytes =225GiB |
| PLE表 | 128 shards，102400491520 bytes，约95.368GiB；每片2500012×160 BF16 |
| 其他PLE参数 | 9 tensors，65679640 bytes；与表、MTP/vision分别记账 |
| 本地工具测试 | 原20项加新增15项，共35项 PASS；包含错误设备、缺失runtime前置、过期遥测、超预算、SMT重复和语义名称别名负例 |
| ARM原生验证 | 最终15组配置、90个token-shape用例，每例3次一致；所有权跳过/恢复、全跳过零输出、incremental无贡献保持值均通过 |
| CPU误差 | 所有最终用例最大relative-L2=0.00020935923718425525，低于预冻结0.01 |
| 完整单层backing | BF16/C64加载512 experts，数值抽查10个实际expert（含0和511）；峰值进程RSS10163636KiB≈9.69GiB |

实际权重选自layer0/23/24/47，主输入是固定seed合成BF16激活；覆盖qlen
1/2/9/10/11/32，单池/C64，BF16与F32存储对照，另含BF16 numerical-compat
模式回归。完整512 backing仅覆盖layer0，**不等于512个expert全部数值执行**。
本轮没有实际模型中间激活、NPU对照、全模型compact或CMMLU结果。

## 2. 修复了什么

原始BF16/C64探针在构造时复现：
`RuntimeError: unsupported unaligned Llamafile intermediate_size`。
640不能整除旧256-block规则；C64需要320+320。

三个原生文件的最小修改：

- `kt-kernel/operators/moe-tp.hpp`：ARM非SVE构建的同类型BF16权重进入32对齐分片。
- `operators/llamafile/tp_slices.hpp`：明确可用BF16/F32混合kernel的架构边界。
- `operators/llamafile/moe.hpp`：BF16 grouped-prefill尾块及FP32 TP partial；不将每份down partial先舍入为BF16。

只打开本次实测的 ARM NEON、BF16-weight/F32-input 路径。主机支持SVE，
但此构建显式关闭SVE/BF16 ISA选项；其他平台/SVE路径和量化格式仍保留原guard。
没有宣称原生BF16 ISA性能或MXFP4/Q8_0/C128资格。

保留第一次RED、编译编辑错误日志和最终GREEN结果；一次编译因变量重命名遗漏
失败，修正后构建通过。没有改变数值门限或丢弃失败来选择结果。

## 3. 实际CPU环境与资源

CPU-only镜像ID：
`sha256:601e76d17e9fb4b1c5a72d491cf086c533a309c5c651315e73fe783c90460c8a`。
Python3.11.15、torch2.9.0；关闭torch设备后端自动加载，未导入torch_npu进行测试。
精确llama.cpp/pybind11 pins取回后应用仓库现有GGML补丁，从Round9代码重建。

最终扩展SHA256：
`f2695802dbd0196aeb644286ce4b02d641200e16b135fb248a008b0ecf1eafbe`。
构建flags、受测头文件与脚本hash见 `cpu-runtime-manifest.json`。

最终测试容器：无NPU设备、network none、非privileged；C64 worker绑定NUMA6/7
各32物理核，cpuset每核一个SMT线程并另列8辅助核。CPU quota16、host memory32GiB、
memory-swap同上；SYS_NICE用于NUMA/线程绑定。此为共享host数值实验，**不是C64
独占性能实验**。容器正常退出0，OOMKilled=false；未改宿主包或训练容器。

本地可复核工具：
`python3 -m unittest discover -s migration_bridge/round9/tests -v`。
ARM probe入口 `migration_bridge/round9/tests/native_unblocked_probe.py`；最终矩阵
命令及参数见证据目录的 `run_final_cpu_probes.sh`。重跑需新结果目录/文件名，
不能覆盖已保存回执。旧F32 TP slice C++回归也已通过。

## 4. 物理15当前阻塞

宿主Card7/chip1/physical15持续训练。61次只读采样历时约637秒：HBM百分比
均为13%（整数读数），AICore0–77%，NPU utilization6–77%。瞬时余量不等于
训练未来峰值已知或已预留显存，不能据此启动Q64全模型。

实际配置与尝试：

- `multi-die-policy = UNION_POLICY`（双Die联合模式）。
- Card7/chip1 `device-share = False`；此前查询无vNPU实例。
- 非privileged独立容器只挂载davinci15及必要管理设备；npu-smi查询返回
  `npu get board type failed. ret is -9005`。没有进一步分配tensor/NPU context。
- 进程内logical→physical映射和共享访问均未通过，准入工具返回BLOCKED。
  回执保留观测原始时间，后续复评也拒绝过期遥测，不以重新写时间伪造新采样。

这证明当前已测试的单Die容器路径不通；不把-9005武断解释为唯一原因，也不声称
所有宿主多进程方式均不可能。对照[openEuler vCANN-RT官方文档](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/ubs-virt/ubs-virt-enpu/vcann-rt/README.html)：多容器共享需开启device-share，A3还涉及单Die直通策略。
该文档列出A3对应CANN8.5/9.1及HDK26.0.0+；当前候选镜像CANN9.0、宿主26.0.rc1，
不能直接当作该软切分组合已受支持。没有安装vCANN、改变全局模式或使用privileged
容器绕过条件。

下一步NPU条件：训练方可提供维护窗口时，先核实实际固件/驱动的单Die直通与
共享支持、变更作用范围及回退条件，再形成具体设置方案；或提供已支持的隔离
访问环境。**本轮授权不包含停止训练或在线改变全局模式。** 已向用户提出窗口
选择，等待其安排；期间允许独立CPU工作，不跨过NPU门禁。

## 5. 源码可重建缺口

当前SGLang gitlink `f02f304a25364d1e8fe9c52515a22f20f5089a52` 从配置的
panxy1019/KT-ASCEND以及kvcache-ai/sglang两个remote均返回 `not our ref`。
.22旧源码实际HEAD为eaa58ccea4ee372b8b207dd16f398eafe3da37ff且含旧runtime改动，
也不能取到该object。未改这些工作树，未将旧overlay冒充精确gitlink。

已独立取回官方KT所锁定SGLang `541ddc37cbc92c60dc748db5ff1a2aad0b069a80`
用于审计；未替换本项目gitlink或把它标成已部署。其原生Qwen4入口仍缺失。
已有干净Qwen4语义快照4ccff141dbe992794f9da6c3aa23535b4f72000d可作为移植对照，
但QSA backend/KV pool、GR、PLE和运行时注册不能仅靠拷贝模型文件完成。
后续须恢复原object，或在可重建base上逐项移植本地KTEP与Qwen4语义并重新测试。

## 6. 当前里程碑

| 项目 | 状态 |
|---|---|
| 下载/复制身份 | PASS，保留既有独立目的端证据 |
| 资源门禁 | CPU局部可执行；NPU BLOCKED_SHARED_ACCESS；全模型未准入 |
| 源码语义 | header结构PASS；SGLang重建/模块语义仍IN_PROGRESS |
| BF16/C64 CPU局部路径 | 本文有界条件PASS；真实模型激活、NPU对照和全48层系统未验证 |
| MXFP4/Q8_0、C128 | NOT_RUN，既有guard/转换实现仍须逐项验证 |
| 模块/全模型/CMMLU/PLE异步/调度/长上下文/正式性能 | NOT_RUN |

独立审查发现准入器接受“无runtime键、只有其他PASS键”的缺陷，已加入失败回归
并修复；额外修复非规范layer/shard名称导致同一逻辑tensor重复计入的问题，
同样经过RED→GREEN。35项工具测试最终通过。没有核心修改的已知重要审查项遗留。

round7定时任务保持暂停；无Round8性能恢复、push、main合并或tag。
