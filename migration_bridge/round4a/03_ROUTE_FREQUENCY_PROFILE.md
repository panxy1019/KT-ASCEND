# Route Frequency Profile

Status: `PASS`.

The all-NPU recorder collected `F[layer, expert]` from Corpus S with greedy
32-token generation. Dimensions and real MoE layer IDs are derived from the
frozen model config. Ranking is deterministic by `(-frequency, expert_id)`.

The A3 run completed 12 requests and recorded a `[1000, 27, 64]`
`logical_count` buffer. Its summed frequency matrix is `[27, 64]`, contains
`80496` routes and has raw recorder artifact SHA256
`cd405edb882b85c8a9651b428a7ec35ba5c3f04cd2bfdb1b6af67948fa1c9a22`.
The frozen ranking and every per-layer count are preserved in
`placements/ranked_experts_per_layer.json`.
