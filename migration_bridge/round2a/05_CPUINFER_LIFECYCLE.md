# CPUInfer / Wrapper Lifecycle

## Status

`PASS` — weight lifetime、1000 forward、20 次 create/load/forward/destroy、不同 fixture、双层 wrapper、1/2 threadpool configuration 均在 A3 通过。

## Weight ownership

实际顺序：

```text
LlamafileMoEWrapper.__init__
  -> GGUFLoader (canonical path aware cache)
load_weights
  -> gate/up/down tensors
  -> weights_to_keep tuple
  -> construct MOE
  -> MOE.load_weights_task
  -> CPUInfer.sync
  -> weights_to_keep = None
forward after Python temporary tensors are gone
```

source：`kt-kernel/python/utils/llamafile.py:LlamafileMoEWrapper.load_weights:203-253`。测试在 load 后释放 Python 临时权重并继续 forward，未出现 use-after-free 或数值变化，说明 MOE 的本地 weight storage 在 sync 完成后独立有效。

## Repeated forward

qlen=1、batch=1、1000 次 wrapper-level forward：

```json
{
  "iterations": 1000,
  "rss_before": 329977856,
  "rss_after": 329977856,
  "rss_delta": 0,
  "rss_samples": [329977856, 329977856, 329977856, 329977856, 329977856,
                  329977856, 329977856, 329977856, 329977856, 329977856]
}
```

结果 deterministic、无 crash，10 个阶段采样没有单调增长。这是明显泄漏检查，不是长期 soak 或 heap proof。

## Loader reuse and create/destroy

执行至少 20 个交替 lifecycle，覆盖：

- same canonical fixture path reuse；
- different fixture path replacement；
- create → load → forward → destroy；
- layer 0 / layer 1 keys；
- garbage collection 后再次构造。

发现并修复 `_gguf_loader_instance` 仅按 singleton 复用导致不同路径拿到 stale keys 的风险。现在 `kt-kernel/python/utils/llamafile.py:LlamafileMoEWrapper.__init__:27-28,77-87` 同时缓存 canonical realpath，路径改变时创建新 loader。

## CPUInfer configuration reuse

原来的单一 CPUInfer singleton 会让先创建的 worker config 污染后续 wrapper。现在 `kt-kernel/python/experts_base.py:BaseMoEWrapper._get_cpu_infer:175-211` 以 `(subpool_numa_map, subpool_thread_count)` 为 cache key：相同配置安全复用，不同 1/2 pool 配置各自持有正确 backend。

覆盖结果：

- 1 subpool：NUMA `[0]`，threads `[4]`；
- 2 subpools：NUMA `[0,1]`，threads `[2,2]`；
- layer 0 / layer 1 不混用 weight key；
- same/different path 不产生 cross-test contamination。

## Evidence

- focused：`logs/round2a/cpuinfer-lifecycle.log`
- combined：`logs/round2a/full-test-matrix.log`
- lifecycle module：5 tests PASS
- total combined：`21 passed in 4.82s`
