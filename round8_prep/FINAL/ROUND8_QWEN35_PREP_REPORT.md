# Round8 Qwen3.5 前置工作报告

> **2026-09-19 执行更新：**此前 prep 快照已被后续实际执行推进。
> 全量 checkpoint 校验、40 层 CPU 转换/身份审计、Q64 全模型短生成、
> 4-prompt 重复/prefix 与 clean restart、40 层 shared composition oracle
> 均已通过；32 个独立 prompt 的双重复生成检查正在后台运行。
> 最新事实以 [CURRENT_EXECUTION](../10_HANDOFF/CURRENT_EXECUTION.md) 和同目录
> `ROUND8_QWEN35_PREP_STATE.json` 为准。以下正文保留历史 prep 状态，
> 不应据此重复下载、重启健康任务或误称 quality 已通过。

日期：2026-09-18。状态：**STATIC_PREP_COMPLETE / RUNTIME_DEPENDENCY_GATES_PENDING**。不是 Qwen A3_VERIFIED，也不是全模型 READY_FOR_RUNTIME。

## 已完成

- 独立 worktree/branch，基于159e3dfcbbda94a13a36d004dde47c1fd9f05374；Round7未改动。
- 模型固定为`Qwen/Qwen3.5-35B-A3B`，revision `59d61f3ce65a6d9863b86d2e96597125219dc754`。配置、tokenizer、generation config、索引、本地元数据SHA和14个分片的Hub声明SHA/大小已保存。分片总71,903,878,016bytes，未下载权重。
- KT与两套SGLang源码固定、对比，未升级生产代码。KT子模块含Qwen3.5，但不能从源码存在推出Kunpeng+Ascend已支持。
- 配置确认40层、256 experts、topK8、hidden2048、intermediate512；30GDN+10full-attention。Vision/MTP存在但初始关闭。
- 确定两条CPU候选：真实ARM probe通过的BF16 native，或保留BF16数值的LLAMAFILE F32。未把x86教程当ARM证据。
- 从配置推导容量，生成Q64/Q32/Q16嵌套均匀mask、映射审计数据和权重索引清单；排除MTP expert条目。
- 实现fail-closed资源准入校验，独立本地venv；7项静态单元测试PASS。测试不是KTEP实例化或kernel运行。
- 准备官方兼容矩阵、issue registry、patch queue、launch模板和Round7交接顺序。

## 关键风险 / 尚未做

1. KT child依赖torch2.9.1/transformers-kt5.6.0.post3，与记录的Round7 ABI不同；隔离Ascendimage和依赖lock尚未完成。
2. KT的独立GDN投影与upstream融合投影不同；#22674为Closed页面，不能假定已合入或直接拷贝。
3. Qwen主专家是packed gate_up/down；原DSV2转换器不能未经验证复用。完整F32 CPU payload120GiB，必须额外预算匿名缓冲/pagecache/转换scratch。
4. 没有下载真实权重、转换GGUF、构建native轮子、运行模型构造、NPU smoke或质量测试；没有创建/启动远端容器。
5. 最终production base等待Round7 final state。资源实时可用性未在本次prep重新采集。

## 容量摘要（ANALYTIC_DERIVED，非实测）

| Profile | 每层NPU experts | routed BF16 payload |
|---|---:|---:|
| Q256 | 256 | 60GiB |
| Q64 reference | 64 | 15GiB |
| Q32 target | 32 | 7.5GiB |
| Q16 optional | 16 | 3.75GiB |

以上不含attention/GDN状态、shared、embedding、KV、workspace、vision和allocator。Q64不称为All-NPU oracle。

## 下一步

先关闭Round7并冻结运行时，再完成隔离依赖/算子probe和真实sample conversion；经资源准入下载/转换后按Q64→Q32顺序运行。无需重做本次模型/源码/静态工具准备；不能跳过仍未执行的runtime gates。

没有push、merge main或tag。当前只提交本地前置资料和工具。
