# GDN canonical source audit

Date: 2026-09-20
Scope: read-only audit of the Qwen4-Exp Gated Delta Network contract inherited
from Qwen3.5 in SGLang `4ccff141dbe992794f9da6c3aa23535b4f72000d`,
compared with the recovered NPU-capable SGLang tree
`f02f304a25364d1e8fe9c52515a22f20f5089a52`. This freezes a minimal,
independent PyTorch reference contract for the text-only, TP1, MTP-off Round9
profile. It does not qualify an optimized kernel, cache manager, model
integration, NPU execution, prefix cache, cancellation path, or full model.

## Source conclusion

Qwen4-Exp uses the Qwen3.5 GDN model path for every `linear_attention` layer.
The local materialization of `4ccff141` is sparse, so several GDN files are not
present as ordinary worktree files. Inspection through `git ls-tree`,
`git grep HEAD`, and `git show HEAD:<path>` confirms that the commit contains the
model, generic backend, packaged recurrence/convolution kernels, cache lifecycle,
and dedicated Ascend backend.

The recovered `f02` checkout remains useful as a simpler older implementation.
On NPU, its generic `GDNAttnBackend` replaces causal convolution and GDN
recurrence calls with `sgl_kernel_npu` operations. Current upstream additionally
has a dedicated Ascend wrapper. The important reusable sources are:

- `python/sglang/srt/models/qwen3_5.py`: checkpoint-facing projections,
  convolution, recurrent parameters, gated normalization, and output projection.
- `python/sglang/srt/layers/radix_linear_attention.py`: dimensions and dispatch
  contract between the model and stateful backend.
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`: prefill/decode
  convolution, packed-sequence boundaries, recurrent-cache update, and NPU
  substitutions.
- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`: decode and
  chunk-kernel arguments, query/key normalization choice, and state indexing.
- `python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py`: FP32 decay and
  update-rate equations in current upstream; f02 carries the older copy under
  `python/sglang/srt/layers/attention/fla/`.
- `python/sglang/kernels/ops/attention/fla/fused_recurrent.py`: direct token
  recurrence, grouped-value-head mapping, FP32 accumulator, and output scale.
- `python/sglang/kernels/ops/attention/fla/chunk.py` plus its `chunk_*` helpers:
  mathematically equivalent chunked prefill and 64-token state boundaries.
- `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`: causal
  depthwise convolution and its rolling raw-input window.
- `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`: exact
  normalization and gate order, including the NPU activation restriction.
- `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`:
  Ascend-specific decode, prefill, target-verify, NPU causal convolution layout,
  recurrent-state writeback, and prefix-state tracking.
- `python/sglang/srt/configs/mamba_utils.py` and
  `python/sglang/srt/mem_cache/memory_pool.py`: cache shapes, dtypes, zeroing,
  allocation, copy, and reuse.
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`: request
  slot mapping and prefix-state tracking at chunk boundaries.

The recurrence below is directly visible in both the f02 and current upstream
kernel sources and is the canonical oracle. Ascend orchestration is present in
the upstream tree, while the device operators it invokes live in the external
`sgl_kernel_npu` package; their equivalence still requires differential tests.

## Frozen checkpoint instance

The local metadata config fixes:

| Quantity | Value |
|---|---:|
| hidden width | 2560 |
| total layers / GDN layers | 48 / 36 |
| GDN layer pattern | layers 0, 1, 2, then every non-`3 mod 4` layer |
| key/query heads and dimension | 16 x 128 |
| value heads and dimension | 48 x 128 |
| value heads per key/query head | 3 |
| query or key projection width | 2048 |
| value or output-gate projection width | 6144 |
| mixed convolution width | 10240 |
| convolution kernel | 4 |
| convolution activation | SiLU |
| output gate | sigmoid |
| RMS epsilon | `1e-6` |
| model dtype | BF16 |
| recurrent state dtype | FP32 |

For each of the 36 GDN layers, the checkpoint index contains exactly these
weights:

```text
linear_attn.in_proj_qkv.weight   runtime shape [10240, 2560]
linear_attn.in_proj_z.weight     runtime shape [ 6144, 2560]
linear_attn.in_proj_b.weight     runtime shape [   48, 2560]
linear_attn.in_proj_a.weight     runtime shape [   48, 2560]
linear_attn.conv1d.weight        runtime shape [10240, 1, 4]
linear_attn.A_log                runtime shape [48]
linear_attn.dt_bias              runtime shape [48]
linear_attn.norm.weight          runtime shape [128]
linear_attn.out_proj.weight      runtime shape [2560, 6144]
```

These shapes are derived from the frozen config and the audited loader; the
local evidence directory contains the checkpoint index, not the tensor shards,
so tensor contents and serialized dtypes were not inspected. The newer model
merges `in_proj_qkv + in_proj_z` into runtime `in_proj_qkvz` and merges
`in_proj_b + in_proj_a` into `in_proj_ba`, with packed loaders preserving the
checkpoint order. f02 keeps the checkpoint projections separate, which is a
useful simpler reference layout.

## Minimal canonical equations

Let `x_t` be the 2560-wide input of one GDN layer at token `t`. All projections
are bias-free. Split the first projection in checkpoint order:

```text
[q0_t, k0_t, v0_t] = x_t @ W_qkv.T
z_t                 = x_t @ W_z.T
b_t                 = x_t @ W_b.T
a_t                 = x_t @ W_a.T

q0_t: [16, 128]
k0_t: [16, 128]
v0_t: [48, 128]
z_t:  [48, 128]
a_t, b_t: [48]
```

Concatenate flattened `q0`, `k0`, and `v0` into `u_t[10240]`. The convolution
is independent for every channel, causal, has no bias, and applies SiLU after
the four-tap sum:

```text
c_t[c] = SiLU(sum(j=0..3, W_conv[c,j] * u_(t-3+j)[c]))
```

Positions before the start of a fresh request are zero. Split `c_t` back into
`q_t[16,128]`, `k_t[16,128]`, and `v_t[48,128]`. The persistent convolution
state is the last three **raw** `u` vectors, before convolution and SiLU, with
shape `[10240,3]` at TP1. Decode must update this window once per accepted token.

For value head `r in [0,48)`, select query/key head
`h = floor(r / 3)`. Compute the gates in FP32:

```text
g_t[r]    = -exp(float(A_log[r]))
              * softplus(float(a_t[r]) + float(dt_bias[r]))
lambda_t  = exp(g_t[r])
beta_t[r] = sigmoid(float(b_t[r]))
```

Thus `g <= 0`, `0 < lambda <= 1`, and `0 < beta < 1` for finite inputs. With
`eps=1e-6`, normalize query and key independently, and scale the normalized
query by `1/sqrt(128)`:

```text
qhat = q / sqrt(sum(q*q) + 1e-6)
khat = k / sqrt(sum(k*k) + 1e-6)
qs   = qhat / sqrt(128)
```

For each value head, persistent state `S[r]` is a `128 x 128` matrix whose axes
are key dimension then value dimension. Starting from `S_(t-1)`:

```text
S_decay = lambda_t[r] * S_(t-1)[r]
error   = v_t[r] - khat_t[h] @ S_decay
delta   = beta_t[r] * error
S_t[r]  = S_decay + outer(khat_t[h], delta)
o_t[r]  = qs_t[h] @ S_t[r]
```

The decay happens before the delta residual is predicted. The output observes
the newly updated state, including the current token. Query and key L2
normalization and the query scale are both required; omitting either changes the
model. Three value heads share one normalized query/key head but have distinct
`v`, `a`, `b`, `A_log`, `dt_bias`, and recurrent state.

The state accumulator and stored temporal cache are FP32. Projection,
convolution, and returned recurrent output may be BF16, but every oracle step
must cast `A_log`, `a`, `dt_bias`, `b`, normalized dot products, decay, error,
outer product, and state accumulation to FP32 before casting the token output.
At TP1 the temporal cache is `[48,128,128]`, or 786,432 FP32 elements (3 MiB)
per request per GDN layer.

The final per-head gated normalization is normalization before gating:

```text
rms_t[r] = o_t[r] * rsqrt(mean(o_t[r]^2) + 1e-6)
y_t[r]   = (rms_t[r] * norm_weight) * sigmoid(z_t[r])
output_t = flatten(y_t) @ W_out.T
```

The RMS reduction is over each 128-wide value head, the learned weight is shared
across the 48 heads, and there is no normalization bias. The frozen sigmoid is
not interchangeable with the recovered f02 default `SiLU(z)` gate.

## Prefill, chunk, decode, and reset contract

A plain eager oracle can process each request sequentially and is the reference
for every optimized mode. Packed execution must preserve exactly the same
request boundaries.

- Fresh prefill starts both caches at zero. A resumed prefill uses the matching
  request's prior convolution window and recurrent matrix only when its prefix
  length is nonzero.
- Packed prefill uses `[1,total_tokens,...]` tensors plus cumulative sequence
  offsets. Neither convolution nor recurrence may cross an offset boundary.
- Chunking at any positive boundary must equal an unchunked run token for token,
  including chunks shorter than the three-token convolution history. Each chunk
  carries the final three raw projection vectors and final FP32 recurrent state
  into the next chunk.
- f02's FLA prefill internally uses 64-token state boundaries. Prefix-cache
  tracking selects the final state for a sequence ending on a boundary and an
  intermediate `h` state otherwise. That bookkeeping is an optimization detail;
  it must be differential-tested against the sequential oracle at lengths
  `1,2,3,4,63,64,65,127,128,129` and across arbitrary external chunk splits.
- Decode consumes one new projected token, rolls the raw-input convolution
  window, applies the recurrence once, and writes both states in place at the
  request's cache slot. Prefill followed by decode must equal one unbroken eager
  sequence.
- Request slots are isolated. Copy/fork must copy both states together. Reorder
  must preserve the request-to-slot mapping. Cancellation/free may leave bytes
  behind, but the next allocation must zero both states before use.
- f02's `MambaPool.alloc` performs the required zeroing; `free` only returns a
  slot to the free list. A replacement pool or bypass of `alloc` therefore
  needs an explicit reset, and `clear` by itself is not evidence that tensor
  contents were erased.

MTP/speculative verification is outside the Round9 profile. Its intermediate
state and tree-convolution code must not be used to qualify ordinary
prefill/decode, and ordinary qualification makes no claim about it.

## Reusable pieces and exact blockers

The f02 projection layout, causal convolution, FP32 gating formula, direct
recurrent kernel, chunk implementation, packed offsets, and Mamba pool design
are suitable sources for an eager oracle and later NPU differential tests. They
are not a drop-in exact Qwen4 implementation for these concrete reasons:

1. **Wrong output gate on the recovered model path.** f02 constructs
   `RMSNormGated` without `activation=config.output_gate_type`, so it uses swish.
   The frozen checkpoint requires sigmoid. Current upstream passes
   `output_gate_type` explicitly, but its
   `python/sglang/kernels/ops/attention/fla/layernorm_gated.py` NPU wrapper still
   asserts that only swish is supported before dispatching to
   `sgl_kernel_npu`. Thus source availability does not resolve the Qwen4 sigmoid
   gate on NPU.
2. **`A_log` construction is not pinned to FP32 in f02.** f02 creates it with
   the ambient default dtype; newer upstream explicitly uses
   `dtype=torch.float32`. Loading may happen to preserve or cast the checkpoint,
   but exact behavior cannot depend on ambient defaults.
3. **NPU recurrence is external.** `causal_conv1d_*_npu`,
   `chunk_gated_delta_rule_npu`, and
   `fused_sigmoid_gating_delta_rule_update_npu` come from `sgl_kernel_npu`, not
   this source tree. Upstream's `ascend_gdn_backend.py` supplies their call and
   state-management contract, but their state dtype, head grouping,
   normalization epsilon, query scale, mutation, and packed-boundary behavior
   still require oracle comparison.
4. **The frozen Qwen4 model/config is not registered in f02.** Reusing GDN
   primitives does not supply Qwen4-Exp model loading, the 36-layer mapping,
   hyperconnections, PLE, QSA, MoE routing, or full-model cache accounting.
5. **TP-dependent cache transfer changed upstream.** Newer upstream records the
   convolution sub-blocks `[key_dim,key_dim,value_dim]` because each is sharded
   independently. Round9 is TP1, so this does not alter the oracle, but f02's
   single contiguous assumption must not qualify later TP or disaggregated
   prefill/decode work.

## Independent reference-test contract

The first implementation should expose a pure eager step taking projections,
`conv_state`, and FP32 `S`, and return the token output plus new states. Required
tests before any optimized-path PASS are:

1. Compare the direct recurrence against a literal FP32 implementation for
   random and adversarial gates, including very negative/positive `a` and `b`,
   zero q/k, and nonzero initial state.
2. Verify the 3:1 value-to-key head mapping with distinguishable per-head inputs;
   a modulo mapping is wrong.
3. Compare full sequence, every-token decode, and varied chunk partitions,
   checking token outputs, convolution state, and recurrent state separately.
4. Verify fresh reset, free/reallocate reset, two-request isolation, batch
   reorder, and state copy/fork. Include one request with identical current token
   but different history so accidental state sharing is observable.
5. Compare sigmoid-gated normalization to the explicit FP32 formula and include
   inputs where sigmoid and SiLU differ strongly, such as negative `z`.
6. Assert temporal state remains FP32 after every prefill/chunk/decode write,
   while the returned activation has the requested model dtype.
7. Compare the available NPU kernels independently for prefill and decode at the
   boundary lengths above. A prefill-only PASS does not qualify decode, and a
   single-token recurrence PASS does not qualify causal convolution history.
8. Load real early, middle, and late GDN-layer tensors only after exact checkpoint
   name, shape, dtype, and count validation. Then compare real-weight eager and
   NPU outputs and both final states. This remains a component result until the
   Qwen4 model and lifecycle integration tests pass.
