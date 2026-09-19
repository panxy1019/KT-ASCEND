# Q64 bring-up failure/fix record

## F1 — language-only is encoder disaggregation, not text-request restriction

Raw `.22/q64-bringup-v1.log`: ServerArgs rejected `--language-only` without
`--encoder-urls`, before model loading. No NPU/model arithmetic failure.
Actual server_args.py `_handle_encoder_disaggregation` and pinned official
[source](https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/server_args.py)
agree (checked2026-09-19). Remove this flag from test launcher only; retain
text-only request protocol, no image/video input and no encoder-disaggregation.
Vision code/weights may be present but multimodal execution is NOT_QUALIFIED.
No production/model/weight changes. Preserve launch-v2; corrected candidate
is launch-v3. Added launch-scope regression; actual parser re-run required.

Resource launch-v1 admission was rejected on32 host swap-out pages; subsequent
fresh launch-v2 admission passed zero-delta window and no owned-cgroup swap.
Do not confuse this earlier admission rejection with F1 software failure.

## F2 — automatic multimodal processor imports unavailable video dependency

Raw `q64-bringup-v2.log`: qwen_vl processor import failed `No module named decord`,
then architecture had no registered processor. `skip_tokenizer_init` would NOT
fix this: tokenizer_manager initializes multimodal processor before that switch.
Use existing programmatic `ServerArgs.enable_multimodal=False`, propagated by
official `ModelConfig.from_server_args`; CLI only exposes store_true, no inverse.
Small `qwen_text_server.py` entry sets the supported field and calls original
run_server, preserving model architecture/weights/tokenizer. No fake decord or
processor stub; no production model changes. Vision execution remains out of
scope. Corrected launch-v4 records explicit programmatic override and entry SHA.
Official source: pinned3424f35d model_config.py and tokenizer_manager.py,
https://github.com/kvcache-ai/sglang/blob/3424f35d0e60f03b9e75d9b90941f8f4cb944524/python/sglang/srt/configs/model_config.py
