# 最小质量 Canary 合同

日期：2026-09-22  
状态：`PROPOSED_NOT_RUN`。本文件只定义未来有独立完整模型预算时的验收；本轮没有追加请求。

## 现有结果的正确命名

现有请求是文本 `Answer with exactly OK.`，`temperature=0`、`max_new_tokens=1`、`ignore_eos=false`。v62 observer-OFF 与 v63 observer-ON 都返回 output ID `[271]`，raw `final_text` 为 `"\n\n"`，finish reason 是 `length`。证据分别为：

- `artifacts/round9-kt-mainline/evidence/v62-base-off.json`，SHA256 `1948965bc1e90fea060da43e136c62a99818dd23c0d0e0e95768a0dc554a181c`；
- `artifacts/round9-kt-mainline/evidence/v63-base-profile.json`，SHA256 `42cb28bf9b195274afbba4b745d6a6d782c07a09094a62dabfa5b6839711b355`。

这一结果应命名为 `PATH_CONSISTENCY_MATCH`：两个观察设置产生相同 token trajectory。它不是 `QUALITY_PASS`，也不能说明模型“答错”的根因。生成上限只有一个 token，且请求没有保存预期 tokenizer tokenization、chat template、允许的前导空白或 stop 条件，因此“输出必须是 OK”不是现有运行可判定的合同。

## 未来 canary 的冻结输入

下一次运行必须使用独立批准的完整模型 request budget，并在启动前冻结以下内容：

1. 模型 revision、config/tokenizer/checkpoint receipt、SGLang/source tree、native extension、CPU bank 和实际 Q32 map；
2. 完整 request body，包括 API endpoint、是否应用 chat template、system/user 消息或 raw text、temperature、seed、`max_new_tokens`、EOS/stop 列表和 streaming 设置；
3. 主配置：C128、MXFP4 CPU、原始 BF16 NPU/NZ29、Static Uniform Q32、Layerwise、streaming32、显式 PLE ASYNC；同时注明 PLE 全局默认仍 OFF；
4. 一个冻结默认路径对照。默认路径必须与候选使用相同请求和模型身份，只有预先声明的候选因素不同；观察器 ON/OFF 可以做无行为回归对照，但不能替代质量对照；
5. 原始响应事件、output IDs、raw text、finish reason、prompt/completion token 数和 request ID。不得只保存 trim 后文本。

## 建议的答案规则

继续使用短、确定、无需外部知识的已知答案。`max_new_tokens=8`、`temperature=0` 可作为预先冻结的首个候选设置，但 8 不是已经证明“足以生成答案并到达 EOS/stop”的保证；tokenizer/template 的离线检查只能确定编码和请求形态，也不能保证模型会在 8 tokens 内停止。预算和 stop 规则必须在运行前固定；若结果仍以 `length` 截断，则判 `INVALID_CONTRACT`，不能把截断结果当质量结论，也不能在看到输出后临时扩大等价规则。

推荐语义判定：对 raw text 只做预先声明的解析——允许去除 Unicode 首尾空白，然后要求解析结果严格等于 `OK`；不忽略标点、不做大小写折叠、不接受附加解释。另保存 token IDs 和未经处理的 raw text。若 tokenizer/template 预检查表明 `OK` 需要不同预算或 stop 设置，应在设备运行前修订合同并记录新版本，不能在看见输出后改答案规则。

结果状态：

| 状态 | 条件 |
|---|---|
| `PASS_KNOWN_ANSWER` | 默认路径和候选都结束于预期 stop/EOS，解析结果均严格为 `OK`，无异常、非有限值或状态泄漏。 |
| `FAIL_SEMANTIC_DIVERGENCE` | 同身份/同请求下，默认路径通过而候选不通过，或两者在预先规定的等价规则外分歧。暂停候选晋级，从首次分歧模块定位。 |
| `FAIL_SHARED_QUALITY` | 默认路径和候选都没有产生已知答案。说明共同质量问题；不能归罪于候选，也不能把一致性写成 PASS。 |
| `INVALID_CONTRACT` | 因 length 截断、错误 template、错误 stop、服务异常或缺失原始证据而无法判断。修正合同后需新预算重跑。 |

## 执行与升级边界

该 canary 必须单独计入 model start、HTTP request、device profile 和 component budget；不得借 shadow decomposition 的组件预算自动执行。先运行默认路径，再在相同已冻结条件下运行候选；若任一身份不一致，整对结果无效。

一个通过的 canary 只提供最小语义护栏，不完成 CMMLU、产品质量、正式性能、长上下文或部署资格。一个可重复的候选专属分歧足以暂停性能候选晋级，但在获得首次分歧证据前，不应从 `"\n\n"` / token 271 推断 GDN、QSA、PLE、Expert bank 或调度是根因。
