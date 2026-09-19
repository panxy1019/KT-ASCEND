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
