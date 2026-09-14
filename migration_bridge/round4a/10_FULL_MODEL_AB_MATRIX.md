# Full-Model A/B Matrix

## P0 replay

- 15/15 requests succeeded.
- 15/15 token sequences exactly matched all-NPU.
- Prefix determinism passed.
- All logprobs finite.
- Maximum `|delta logprob|`: `0.08105409145355225`.

## P1 held-out Corpus V

- 45/45 requests succeeded.
- 30/45 token sequences exactly matched all-NPU.
- Both all-NPU and Hybrid prefix determinism passed.
- All logprobs finite.
- Maximum `|delta logprob|`: `2.4378519617021084` at
  `v_struct_03`, 64-token case, step 51.
- First divergence examples: `v_en_01` step 1, `v_struct_02` step 6,
  `v_struct_03` step 9, `v_zh_02` step 19 and `v_en_03` step 31.

P1 is therefore blocked by mandatory token and logprob gates. P2/P3 A/B
matrices were not run.
