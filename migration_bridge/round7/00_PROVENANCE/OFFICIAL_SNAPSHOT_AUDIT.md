# Official scheduler snapshot audit — CODE_INSPECTED

2026-09-17: fetched official KT main
`0b954fe4ca2f9e79519da66156e6a3f92b686232`. Its gitlink is still
`3424f35d0e60f03b9e75d9b90941f8f4cb944524`.
Fetched official SGLang main `541ddc37cbc92c60dc748db5ff1a2aad0b069a80`.

Comparing pinned child with latest child: `kt_ep_wrapper.py` changes are RAWINT4
SFT loading/support and a vendored compressed-tensors import; this diff does not
supply a scheduler migration or generation-publication fix. No latest-main merge
is justified by that comparison. Server-args changes are recorded separately
before considering any adapter. The deployed child is neither audit snapshot:
actual deployed Python hashes take precedence.

The official `generate_gpu_experts_masks` flattens `[layer, expert]` and chooses
global top-K. Round7 routed budget is 208. Dense layer0 must have zero frequency
before selection; KT restores its dense mask after routed selection.

Actual deployed stat accumulator supports buffer-size <0 as an infinite buffer;
positive size uses a circular buffer. Round7 uses -1 and verifies count totals,
rather than trusting dump-file existence as proof of full-corpus collection.

Actual BF16 Dynamic copies source logical expert rows into resident slots with
Tensor.copy_, then updates mask, int32 mapping, inverse mapping and wrapper mask.
There is no explicit generation audit in this code. Serial model execution and
stream ordering still require runtime validation; inspection alone does not
establish I16 or migrated weight identity.

Official `select_top_experts_from_batch` uses torch.topk. Its comment promising
lower-ID tie preference is not a specification guaranteed by torch.topk. A
stable explicit tie reference and regression are prepared; this is not yet a
production patch or an A3-verified defect.

Sources:
[global masks](https://github.com/kvcache-ai/ktransformers/blob/0b954fe4ca2f9e79519da66156e6a3f92b686232/kt-kernel/python/experts_base.py),
[pinned wrapper](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/layers/moe/kt_ep_wrapper.py),
[latest wrapper](https://github.com/kvcache-ai/sglang/blob/541ddc37cbc92c60dc748db5ff1a2aad0b069a80/python/sglang/srt/layers/moe/kt_ep_wrapper.py).
