# QSA canonical source audit

Date: 2026-09-20
Scope: read-only audit of SGLang `4ccff141dbe992794f9da6c3aa23535b4f72000d`
against the recovered Round6 SGLang pin
`f02f304a25364d1e8fe9c52515a22f20f5089a52`. This defines an independent,
eager PyTorch reference contract for the frozen text-only, TP1, MTP-off Round9
profile. It does not qualify an optimized kernel, model integration, an NPU
backend, or full-model execution.

## Source conclusion

The recovered `f02` tree has no Qwen4-Exp model/config, QSA indexer, QSA sparse
backend, or QSA-private cache implementation. Its similarly named
`attention/compressed/indexer.py` and NSA indexers are different model paths and
must not be used as QSA semantics.

The complete mathematical definition available in the audited `4ccff141` tree
is split across these files:

- `python/sglang/srt/layers/attention/qsa/qsa_indexer.py`: projection split,
  index-query normalization/RoPE, raw-key compression, prefill/decode selection.
- `python/sglang/srt/layers/attention/qsa/mqa.py`: the exact weight-free index
  score, packed prefill mask, and paged decode score.
- `python/sglang/srt/layers/attention/qsa/kernel.py`: FP32 key averaging,
  compressed-block top-k, block-to-token expansion, unfinished-tail inclusion,
  and device-independent final sparse-GQA reference.
- `python/sglang/srt/layers/attention/qsa/metadata.py`: packed row ranges,
  compressed decode view, pending-ring addressing, and first-member RoPE
  coordinates.
- `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`: request-row
  mapping, logical-to-physical cache mapping, compression write plans, and the
  final sparse-attention call.
- `python/sglang/srt/models/qwen4_exp.py`: QSA is run from the pre-attention
  hidden state in parallel with the ordinary Q/K/V projection; its logical token
  selection is then passed to ordinary attention over the ordinary K/V cache.
- `python/sglang/srt/models/qwen3_5.py`: ordinary Q/K/V normalization, RoPE,
  shapes, scale, output gate, and output projection inherited by Qwen4-Exp.
- `python/sglang/srt/layers/attention/qsa/config.py`: supported compressed-QSA
  invariants.
- `python/sglang/srt/mem_cache/qsa_kv_pool.py`: the concrete
  `QSATokenToKVPool`, per-request pending-key/RoPE ring, compressed-key cache,
  setters/getters, page-ratio validation, and memory accounting.
- `python/sglang/srt/mem_cache/kv_cache_configurator.py`: construction of the
  QSA-aware hybrid KV pool for the compressed Qwen4-Exp profile.
- `python/sglang/srt/model_executor/pool_configurator.py`: QSA cache-capacity
  pricing used by runner pool sizing.

The local materialization of `4ccff141` is sparse. A worktree-only search missed
the pool files, but `git grep HEAD` and `git show HEAD:<path>` confirm that the
commit defines `get_qsa_key_state_buffer`, `get_qsa_compressed_k_buffer`,
`qsa_rope_position_buffer`, and `qsa_compressed_page_size` in
`qsa_kv_pool.py`. The commit therefore contains the QSA-private cache integration
as well as the equations. The recovered `f02` tree still lacks those Qwen4/QSA
components, so copying only its older attention primitives is insufficient.

## Frozen checkpoint instance

The verified local metadata config fixes:

| Quantity | Value |
|---|---:|
| hidden width | 2560 |
| model attention heads / KV heads / head dimension | 24 / 2 / 256 |
| index query heads / KV heads / head dimension | 4 / 1 / 128 |
| index token budget | 2048 |
| compression ratio | 4 |
| compressed-block top-k | 512 |
| returned fixed token-index width | 2051 |
| RMS epsilon | `1e-6` |
| main rotary dimension | 64 (`256 * 0.25`) |
| QSA rotary dimension | the same reused 64, within index dimension 128 |

Layers 3, 7, ..., 47 are the twelve full-attention/QSA layers. The checkpoint
index contains, for each of them:

- `self_attn.indexer.index_qk_proj.weight`
- `self_attn.indexer.q_layernorm.weight`
- `self_attn.indexer.k_layernorm.weight`

The projection output width is `(4 + 1) * 128 = 640`. QSA reuses the layer's
ordinary RoPE instance; it does not define an independent position convention.
For this checkpoint the reused RoPE is interleaved three-axis MRoPE with sections
`[11, 11, 10]`. A canonical reference should call the same frozen RoPE routine,
or validate an independent reproduction of its axis composition separately.

## Minimal canonical equations

Let `x_t` be the pre-attention hidden state at logical token position `t`,
`W_idx` the index projection, `H=4` index query heads, `D=128`, and compression
ratio `R=4`.

1. Project once without bias and split in output-feature order:

   ```text
   z_t = x_t @ W_idx.T
   q_raw[t] = z_t[:H*D].reshape(H, D)
   k_raw[t] = z_t[H*D:].reshape(1, D)
   ```

2. Gemma RMS normalization is per 128-wide head in FP32:

   ```text
   gemma_norm(u, w) =
       (u.float() * rsqrt(mean(u.float()**2) + eps) * (1 + w.float()))
       cast back to u.dtype
   ```

3. Normalize each index query head, then apply the layer's RoPE at the query's
   RoPE coordinate. Only the first 64 dimensions rotate for this checkpoint;
   dimensions 64:128 pass through unchanged:

   ```text
   Q_idx[t,h] = RoPE(gemma_norm(q_raw[t,h], w_q), rope_position[t])
   ```

4. A compressed group exists only after all four raw index keys in logical
   positions `4g..4g+3` exist. Average the **raw, unnormalized, unrotated** keys
   in FP32, cast to their storage dtype, then normalize and rotate. The RoPE
   coordinate is the first member's coordinate, not the last or the arithmetic
   mean:

   ```text
   pooled[g] = cast(mean_fp32(k_raw[4g:4g+4]))
   K_idx[g,0] = RoPE(gemma_norm(pooled[g,0], w_k), rope_position[4g])
   ```

5. The weight-free index score for query `t` and completed group `g` is:

   ```text
   score[t,g] = sum_h ReLU(dot_fp32(Q_idx[t,h], K_idx[g,0])) / sqrt(D)
   ```

   ReLU is applied independently to each head's dot product before the head sum.
   There is no softmax, learned per-head weight, bias, or ordinary attention
   scale in this index score.

6. Causal visibility is expressed as a half-open range of completed groups:

   ```text
   visible_completed_groups(t) = [0, floor((t + 1) / R))
   ```

   In packed multi-sequence prefill, each row scores only its own sequence's
   packed compressed range. Top-k returns indices relative to that sequence, not
   global packed-cache columns. Select the highest 512 groups, or every visible
   group when fewer than 512 exist; unused columns are `-1`.

7. Expand each selected group `g` in selected-score order to logical tokens
   `[4g, 4g+1, 4g+2, 4g+3]`. Keep at most 2048 expanded entries. Then append the
   causal, not-yet-compressed tail:

   ```text
   tail_start = floor((t + 1) / R) * R
   tail = [tail_start, ..., t]
   ```

   The tail has 0 through 3 entries. The fixed result width is
   `2048 + R - 1 = 2051`; valid logical indices must be contiguous before `-1`
   padding. Indices outside the sequence length are invalid. Completed groups and
   the tail are disjoint by construction.

8. Convert selected logical indices to physical ordinary-KV slots with the
   selected query row's request mapping. An index is valid only when
   `0 <= index < that row's sequence_length`; otherwise its slot is `-1`.

9. Final sparse attention uses the model's ordinary projected, Q/K-normalized,
   RoPE-applied Q and K and ordinary V. It does **not** attend over `K_idx`:

   ```text
   Q_main: [rows, 24, 256]
   K_main/V_main cache: [physical_slots, 2, 256]
   repeats = 24 / 2 = 12
   logits[h,j] = dot_fp32(Q_main[row,h], K_main[slot_j,kv(h)]) / sqrt(256)
   P[h,:] = softmax(logits[h,:])
   O[row,h] = sum_j P[h,j] * V_main[slot_j,kv(h)]
   ```

   A row with no valid selected slots returns zeros. Causality comes from the
   selected logical indices; the eager reference applies no second causal mask.
   The inherited sigmoid attention-output gate and output projection occur only
   after this sparse attention and are outside index selection.

## State and layout contract

Three QSA-private state structures are distinct from the ordinary K/V cache:

1. A per-request pending raw-index-key ring of four slots, addressed as
   `request_pool_index * 4 + logical_position % 4`.
2. A matching RoPE-position ring. Plain positions are stored as three repeated
   coordinates; MRoPE stores the actual three-axis coordinate.
3. A per-layer compressed index-key cache. A completed group's slot is derived
   from its first ordinary token slot divided by four. Full KV pages must be
   aligned to the compression ratio.

Decode writes a compressed group exactly when the new sequence length is a
multiple of four. Extend writes every newly completed group and requires the
shared prefix length to be group-aligned. The optimized extend path reads group
members directly from the current packed chunk; decode and eager speculative
paths read the four members from the pending ring. These paths must produce the
same compressed key.

Packed prefill rows use:

- `sequence_lengths`: one value per request sequence.
- `token_to_batch_idx`: one request-row id per semantic query token.
- packed compressed keys concatenated in request order.
- `row_start`: cumulative completed-group count before that request.
- `row_end = row_start + min(floor((t+1)/4), floor(sequence_length/4))`.

Decode uses a compressed page table and compressed length
`floor(sequence_length/4)` per query row. Page-table entries after that length
may be stale and must never be read.

For normal decode the physical logical position is `sequence_length - 1`.
Speculative modes distinguish the physical paged-KV logical position from the
model's RoPE coordinate. Round9 currently excludes MTP, but the reference API
should still keep `logical_position` and `rope_position` as separate arguments
so their meanings cannot be silently conflated.

## Independent reference-test contract

The first portable implementation should be an eager PyTorch oracle with plain
per-request lists/tensors. It should not reproduce paged addressing, Triton,
TileLang, CUDA graphs, fused preparation, or scheduler padding until the semantic
tests below pass.

Required tests:

1. **Projection and checkpoint identity:** load the three real indexer tensors
   from an early, middle, and final QSA layer; assert exact shapes and the
   query-first/key-last split.
2. **Normalization order:** compare against independent FP64 equations and prove
   that `mean -> K norm -> RoPE(first member)` differs from normalizing or rotating
   members before the mean.
3. **Score equation:** use head dot products with mixed signs to distinguish
   `sum(ReLU(dot_h))` from `ReLU(sum(dot_h))`; check division by `sqrt(128)` and
   absence of a softmax or head weights.
4. **Causal boundaries:** cover positions 0, 2, 3, 4, 6, 7, and 8. In particular,
   position 2 selects only tail `[0,1,2]`; position 3 can select completed group
   0 and has no tail; position 4 selects completed groups plus exact tail `[4]`.
5. **Top-k and padding:** cover zero completed groups, fewer than 512 groups,
   more than 512 groups, invalid `-1` padding, and fixed width 2051. Avoid exact
   score ties when asserting order because backend tie order is not a semantic
   guarantee; for tie cases compare the valid selected set or final attention.
6. **Packed request isolation:** interleave query rows from two requests with
   different lengths and scores. Verify relative block indices never select the
   other request's packed keys and logical-to-physical conversion uses the
   correct request row.
7. **Chunk/decode equivalence:** feed the same token stream as one aligned
   prefill, multiple aligned chunks, and tokenwise decode. Assert identical raw
   rings, completed compressed keys, logical selections, and final sparse output.
8. **Main-cache separation:** perturb compressed index keys while holding the
   selected logical set constant; only selection may change. Once the set is
   fixed, final attention must depend exclusively on ordinary Q/K/V. Conversely,
   perturb ordinary cached K/V without recomputing selection and require the
   final output to change.
9. **Sparse GQA:** use 24 query heads and 2 KV heads, verify the 12-head repeat
   mapping, FP32 score/softmax/value accumulation, scale `1/sqrt(256)`, `-1`
   removal, and zero output for an empty valid set.
10. **State lifecycle:** two requests with identical positions but different
    request slots must not share pending keys, compressed keys, or physical K/V.
    Reset/cancel/reuse must not expose the prior request's ring or page-table
    entries within the new request's valid length.
11. **MRoPE:** compare QSA query/key rotation with the exact layer RoPE for both
    one-axis text positions and explicit three-axis positions; verify only the
    first 64 of 128 index dimensions rotate and compressed keys use the first
    member's coordinate.
12. **Failure cases:** reject malformed head layouts, `indexer_kv_heads != 1`,
    ratio below 2, budget not divisible by ratio, mismatched row mappings,
    misaligned extend prefixes, logical indices beyond sequence length, and
    query/K/V head dimensions that do not match.

Passing these tests qualifies only the canonical QSA math and request-local state
contract. Upstream supplies QSA-private pool allocation and ordinary-KV mapping,
but Round9 must still exercise their reset/cancel/reuse ownership and the backend
that consumes the 2051-wide logical selection. Dense attention is not a valid QSA
fallback because it changes both the selected context and resource envelope.

## Executed local gate

2026-09-20: CPU6 tests PASS. Physical15 device-v2 passes30 real-weight cases
(layers3/23/47, FP32/BF16, token lengths1/3/4/9/17),3 identical repeats each.
Independent FP64 ordinary-attention/output-gate/projection equations plus
independent real index-query and compressed-key equations. Frozen gates
FP32=1e-4/BF16=.03; all reported errors explicitly finite. Maximum errors:
output1.07935e-6/.00511660; index query4.92099e-7/.00256631; compressed key
5.10292e-7/.00334799 (FP32/BF16).

Controlled520-block test selects exactly the highest512 plus3 tail tokens;
signed multihead fixture distinguishes sum(ReLU(dot)) from ReLU(sum(dot)).
Causal boundaries and tokenwise selection checked. Peak209398784bytes, final
allocated/reserved0. Source identity verified; isolated logical0→physical15.

Evidence: evidence/physical15/20260920-qsa/device-v2.json and frozen run source
.3:/mnt/data1/kt-round9/run-sources/qsa-v2. PASS_QSA_LOCAL_ONLY: no paged-pool,
packed-request, cancel/reuse, multi-axis/VL or full-model qualification.
Short real-weight sequences retain all causal blocks; long-budget ranking
uses controlled synthetic indices, not long real-model activations.
