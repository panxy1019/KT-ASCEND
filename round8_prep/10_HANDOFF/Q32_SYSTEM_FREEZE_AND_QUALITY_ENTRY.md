# Q32 system evidence freeze and quality entry — 2026-09-19

Q32-C64 Static uses the original frozen 32/256 resident profile in all 40 MoE
layers. The profile SHA256 is
`b358399e6405d60b81d8231529b1ec6024ff7138cddab2e0cbac3ccf0b25f486`.
Model, tokenizer, F32 GGUF, LLAMAFILE C64 (two 32-worker subpools), TP1,
batch1, NPU2, and disabled Graph/Dynamic/Deferred remain as in Q64.

Raw Q32 Process A and Process B evidence is preserved in
`q32-system-evidence-v1.tar.gz`, SHA256
`6c7ae7a527bc94ab02411fb9db7aab5381ebdc7cac8f3a07711f1901ec53e21f`.
Independent archive audit found:

- Process A: 4 prompts, 32 requests, exact repeat and 8/16/32 prefix checks.
- Clean restart: 4/4 32-token trajectories exactly match Process A.
- Independent screen: 32 prompts, 64 requests, 32/32 exact repeats, finite
  output logprobs and nonempty outputs.
- Process B structure: 40/40 layers have exactly 32 NPU slots and the shared
  expert composition oracle passes.

These are system checks. Q32 quality and formal performance remain unqualified
at this freeze point. The original Qwen CMMLU manifest SHA256 is
`365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d`.
The audited Q64 engineering reference is 222/256 with zero invalid outputs.
Under the previously frozen observed −5 percentage point rule, Q32 needs at
least 210/256 correct with zero invalid outputs. Q32 uses the same ordered 256
items, token IDs, stop sequences, one-token greedy protocol and A/B/C/D scorer.
Its result is written to a new directory and independently audited after run.
