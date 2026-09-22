# Git and evidence freeze

Branch: `codex/round6-5a-static-numerical-rootcause`.
Base: `b2e9d8721911402e94cf3b103f3e0d3800fd81be`.
Evidence commit: `79f022cb0ecc6a0982aa68d96ebd386b96a2b71f`.
The subsequent documentation-only closure commit is identified by Git history;
this file does not attempt a self-referential commit hash.

| Commit | Concern |
|---|---|
| 847f0c079 | Scope and resident-layout probe |
| 860f4582a | Layout evidence, canonical and causal tools |
| d6b287721 | Ordered causal sequence and arithmetic audit |
| 9e9ea06b0 | Causal decision, prefill extension, evidence tooling |
| 79f022cb0 | Raw/compact evidence and local freeze validator |

Every committed changed path relative to the base is under
`migration_bridge/round6_5a/`. Production source and submodule gitlinks were not
changed. Existing uncommitted `third_party/llama.cpp` state remains untouched.
No push, main merge, promotion tag or historical tag movement was performed.

## Hashes

- Compact target archive:
  `62570988b5f7052beace0df7bb8ba1a4cb3023dab07106206a7f004df0e8c5d2`.
- Full artifact manifest:
  `0b186423eb8939fc029d909013f3889c72c9c9b9bf41a77f1fb5d1c2482237f8`.
- Machine-readable root-cause decision:
  `2b73d85ccac8b0b2658e9375b201ef1ddb34c8398707d1ca8cfd9a5da786270a`.

115,946 target files were hashed. 172 compact artifacts are verified locally;
115,774 raw files are hash-indexed but retained on the target, not copied into
Git. The manifest itself is included and protected by the archive hash above.

Validation:

```bash
python3 migration_bridge/round6_5a/tools/validate_freeze.py
python3 migration_bridge/round6_5/tools/validate_blocked_freeze.py
```

Both checks passed; neither is an assertion that Round6.5 qualification passed.
The dedicated container was stopped after evidence extraction; NPU0 showed no
process. Persistent artifacts, container and image remain recoverable.
