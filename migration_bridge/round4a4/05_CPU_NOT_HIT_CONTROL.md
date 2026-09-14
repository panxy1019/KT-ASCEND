# CPU-Not-Hit Exact Control

Evidence state: **A3_VERIFIED**.

Default-off child instrumentation records the already-computed pre-merge NPU
partial (`gpu_output`) alongside the existing real Layer17 hidden state,
TopK IDs/weights, CPU partial and merged output. No routing or arithmetic is
changed.

Four real 64-token Hybrid generations produced 130 unique hidden-state rows
whose TopK contained none of the frozen CPU experts `{6,8,25,36}`. The first
32 unique rows were frozen before inspection of their equality result.

| Exact condition | Result |
|---|---:|
| real unique eligible rows | 130 |
| frozen verified rows | 32 |
| CPU partial equals all-zero tensor | 32/32 |
| merged BF16 output equals pre-merge NPU partial cast | 32/32 |

This is a direct exact control over the Hybrid merge boundary. It does not use
sampling-only route counts from an already-diverged KV history and does not
infer the NPU partial by subtracting CPU output.

Evidence: `evidence/wp1-cpu-not-hit.json`, SHA256
`60284e3700b1ea1d0a444909852ed733ee631abdd475466f45a49cdcd8d92d50`.

`CPU_NOT_HIT_CONTROL = A3_VERIFIED_READY`.
