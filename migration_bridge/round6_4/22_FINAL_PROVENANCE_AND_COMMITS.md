# Round6.4 最终来源、复核与提交

日期2026-09-16。最终采集及分析在09:44UTC前完成；最终检查模型和队列进程均已退出。

## 实际执行环境（不能用本地文档HEAD代替）

- Destination：110.129.0.22，Ascend910B3，Kunpeng920，CPU0–31，NPU0。
- Container：kt-r64-910b-migration。
- ContainerID：6d997f15ef460a16ac88354b405988a99bf9adcb5e6900a748c4f12c43ec216a。
- Parentcheckout：e3084d309d6f39b8a9f7185d36ac27e4b77526d7。
- SGLangcheckout：eaa58ccea4ee372b8b207dd16f398eafe3da37ff。
- 两者均带显式记录overlay；15共同文件哈希三模式一致。
- Python3.11.15/GCC11.4.0，torch2.9.0+cpu，torch_npu2.9.0.post2。
- Linux6.8.0-107-generic/aarch64，driver25.5.1。
- CANN9.0目标910B算子包安装来源/哈希见21文档。旧metadata生成器的
  `cann_version_files`为空，保留原状；不伪造为当时已采集到完整版本文件。

## 冻结输入及关键产物

| 项目 | SHA/revision |
|---|---|
| 模型revision |604d5664dddd88a0433dbae533b7fe9472482de0|
| tokenizer.json |41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669|
| D1placement |23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c|
| D1GGUF |4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f|
| HD3 |4e54ac5c0e6ab9f25991ca29d54e32d1632d4e687f72d344430d4b5691b8590e|
| FD3 |9b8703afbdfd3bd925096725168f9d68914bd116e726336a972f5376443f7fe9|
| 源平台质量执行协议 |19df819323339c50ae495a5658e89f8cbef2dcb5d8881a00e9de9973ea148184|
| 源平台质量结果 |03cf07513d49b0cf51c8d148192e3a5e1a7e3cb18f3a971280592e50683b37cc|
| 新平台CPU扩展 |0f3aac9009931b18853c9b8fd6f6bba8b02d6ea396e79b2029bf9ce092049895|
| 最终采集元数据归档 |75db40e4cef0308d06eb05971bf81839815c853531f6fbe7379f1acfc187324a|

元数据归档在远端和本地SHA相同；原始PT仍保存在新服务器，不删除。
完整元数据文件哈希表在`evidence/final-descriptive-910b/final-audit.json`。

## 最终复核

- 三模式独立元数据审计PASS；含lookahead边界、同历史、索引、源码、正常回收。
- 原平台质量OFF/ON各288原始请求重新离线审计PASS；重算paired summary字节哈希
  与原冻结结果相同。未重新运行模型或选择新样本。
- 本地工具测试79/79PASS，包含最终归档正例和unknown-forward拒绝负例。
- 最终源码/文档diff检查通过；原始日志不为消除空白警告而改写。
- 用户已有`third_party/llama.cpp`修改保持不动、不提交；不是声称全工作树clean。

## 本次续作主要提交

| 提交 | 内容 |
|---|---|
|817f4b2e4|显式目标CPU构建资格与32worker迁移工具|
|328f2b441、aface70a6|首次失败、CPU修复和构建来源|
|6af2a4518|非目标源码漂移拒绝测试|
|65e64dc58|新平台16worker桥接与真实专家PASS|
|d35a8ec40|绑定新容器的只读采样|
|193b49742、9eccb6b96|32worker桥接及编译源码overlay|
|db691e99c|用户暂停监控，保持后台串行执行|
|c14053c5090a5d022cd8580ef7bbd0b08f557f43|完整三模式证据与最终审计测试|

最终ADR/contract/summary单独docs提交；从git log解析其完整SHA。
本次未push、merge或创建加速READYtag。代码本地提交不等于GitHub已同步。
