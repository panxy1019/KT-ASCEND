# DeepSeek-V2-Lite All-NPU Baseline

## Configuration

- A3 NPU 0 only
- TP = 1
- BF16
- eager execution / CUDA graph disabled
- context length = 512
- max total tokens = 512
- one running request
- no KT wrapper, no CPU expert

## Results

- Model load: PASS
- Prefill: PASS
- Decode 1/8/16/32/64: PASS
- Greedy generation 16/32/64: PASS
- English, Chinese, and structured-numeric prompts: PASS
- NaN/Inf: none
- Deadlock/crash: none
- NPU process memory after load: approximately 29.48 GiB

The saved A/B baseline is
`/home/admin/kt_round3_5b38208/logs/round3/full-model-all-npu-router-matrix.json`.
It is the token/logprob reference for the final Hybrid comparison.

