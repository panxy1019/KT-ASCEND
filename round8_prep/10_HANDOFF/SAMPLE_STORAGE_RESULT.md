# Layer0/E0 storage sample — 2026-09-18

Both pinned shards completed in v4 after301s. Full-file SHA matches the frozen
manifest for shard6 and shard12. Safetensors header audit confirms BF16 packed
gate_up `[256,1024,2048]`, down `[256,2048,512]`; payload offsets are valid.
Raw remote audit `sample-shard-audit-v1.json` is not a production operator result.

Executed `tools/extract_sample_expert.py` in isolated Round8 container with the
read-only reference Python environment. Extracted only text Layer0/expert0;
no MTP and no full tensor materialization. F32 expansion, serialized reload,
and conversion back to BF16 are exact. Finite checks passed.

```text
status = BF16_F32_STORAGE_ROUNDTRIP_PASS
bytes = 12583080
sha256 = a06d8deeb33d02ea5bd651b6f5e5c52175b556a9e28c8995bf0fe81de1bb8f7c
production_gguf = false
operator_qualified = false
gate_up_split = NOT_YET_QUALIFIED
```

Remote artifact `/home/admin/kt-round8-qwen35/sample-expert-storage-v1/`
contains F32 safetensors and summary; matching `.log` is retained. No full
conversion started. Next: verify split/orientation against frozen loader,
canonical/NPU same-operand reference, actual LLAMAFILE512/C64 and GDN execution.
Native BF16 support remains a capability question; no backend selected by CLI
acceptance alone. Full-model Q64/Q32 and quality remain NOT_RUN.
