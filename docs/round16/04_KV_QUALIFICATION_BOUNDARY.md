# KV 资格边界：为什么单页 `unsqueeze(0)` 不构成闭环

## 1. 风险

完整模型早期接线中，单页缓存可能从预期的四维 view 退化为三维 view。添加 `unsqueeze(0)` 可以恢复遗漏的 page 轴，但它只回答一个局部 shape 问题：

```text
[page_size, heads, width] -> [1, page_size, heads, width]
```

它不能证明：

- 多个 page 的逻辑到物理映射正确；
- 非恒等 block table 正确；
- 两个 KV head 都被写入；
- GQA query head 到 KV head 的映射正确；
- page 边界附近位置正确；
- sparse attention consumer 读取了正确数据；
- 完整模型的长上下文状态连续性正确。

所以本轮将 `unsqueeze(0)` 定义为 **local layout compatibility**，明确禁止把它当成 KV 完整资格。

## 2. 多页真实设备 probe

独立 NPU probe 使用：

| 项目 | 值 |
|---|---|
| cache shape | `[3, 3072, 2, 256]` |
| block table | `[[2, 0, 1]]` |
| query heads | 24 |
| KV heads | 2 |
| touched pages | `[0, 1, 2]` |
| logical positions | 12 个，覆盖页首、页尾和跨页位置 |
| K/V store | exact |

非恒等 block table 强制逻辑 page 0/1/2 映射到物理 page 2/0/1。探针不仅比较写入后的 K/V storage，还让 sparse GQA attention 消费这些 cache 行并与独立 reference 比较。

结果：

- relative L2：`0.0018465227913111448`；
- max absolute error：`0.015625`；
- K/V 写入 exact；
- 所有三页均实际触达。

原始结果：[multi-page result](../../artifacts/round16-b/evidence/20260929-round16-b-qsa-multihead-v001/result.json)。

## 3. 多头 store

冻结 Ascend store primitive 每次消费一个 KV head。本轮 adapter：

- 校验 cache 是 `[pages, page_size, kv_heads, width]`；
- 校验输入 row 是 `[rows, kv_heads, width]`；
- 对每个 head 创建 stride/view slice；
- 每个 K/V head 各写一次；
- 不复制完整 cache；
- head 数或 row tail 不匹配时，在写状态前失败。

测试还覆盖两 KV head、HND/NHD view、无效三维 shape 和重复安装。

## 4. 完整模型证据

完整模型通过：

- 五类 exact canary；
- 64 输入／32 输出多步 Decode；
- finite logprobs；
- cancel 后无 unfinished request；
- same-process reuse exact。

但完整模型本轮配置为 `max_model_len=256`，输入和输出各不超过 128。它没有自然运行一个跨多个 3072-token page 的长上下文请求。

因此资格必须分开写：

| 范围 | 状态 |
|---|---|
| 多页 cache mapping/store/attention 组件 | PASS |
| 多 KV head 与 GQA reference | PASS |
| 完整模型 context≤256 | PASS |
| 完整模型长上下文跨页生成 | NOT RUN |
| 广泛 KV correctness | NOT CLAIMED |

机器可读合同：[ROUND16_B_KV_QUALIFICATION.json](../../artifacts/round16/ROUND16_B_KV_QUALIFICATION.json)。

## 5. 后续若扩展上下文需要补什么

任何后续长上下文资格至少需要：

1. 完整模型的真实多页 Prefill；
2. page 边界前后多个 Decode step；
3. 非恒等 block allocation 和 slot reuse；
4. cancel 后跨页 slot 回收；
5. 同进程新请求复用旧物理 slot；
6. QSA index、GDN/PLE state 与 KV generation 一致；
7. 资源峰值和 allocator 回收；
8. 独立 reference 或 teacher-forced earliest-divergence 定位。

在这些证据完成前，不应把本轮组件 probe 外推为 8K、16K 或更长上下文资格。
