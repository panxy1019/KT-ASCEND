# KTransformers Round 3 Commits

Base: `5b382086940f3c5f92bf52120fc1004e6d53026b`

| Commit | Purpose |
|---|---|
| `0893eba` | Use exact ACL stream synchronization and owned synchronous ACL transfers for Ascend CPU expert staging; add decode race regression |
| `94c8bee` | Add DeepSeek-V2-Lite 1408-dimension LLAMAFILE TP1 tail handling and sparse CPU expert loading |
| `d12ffcd` | Advance SGLang submodule to the verified Ascend KT EP child |
| `0d1fbde` | Add real-model export, placement, A/B, identity, layer, and stability tools |

Documentation is committed separately so executable changes, submodule movement,
verification tooling, and audit narrative remain reviewable.

