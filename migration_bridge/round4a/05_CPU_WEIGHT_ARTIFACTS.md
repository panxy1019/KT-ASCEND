# CPU Weight Artifacts

P0 and P1 reuse the frozen Round 3 F32 GGUF containing all 64 routed experts
for Layer 17:

```text
SHA256 a16a50827ec81b54195bf246c7f9d05f7c1d5f3601ee33426c732f65892e180f
```

`tools/export_multilayer_gguf.py` adds deterministic `--layers` support and a
manifest containing the model revision, config/index hashes, source shard and
tensor hashes, shapes, exporter hash and GGUF hash. P2/P3 artifacts were not
generated because P1 failed its mandatory full-model gate; proceeding would
violate the ordered stop rule.
