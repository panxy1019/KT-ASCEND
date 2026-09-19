# Decisions and provenance

2026-09-19 user route accepted: Round8 closure → Qwen3.8 Q64/F32 → low-bit →
PLE async/shared budgets → scheduler/Q32 → context tiers. No Qwen3.6.
Deferred OFF/noncritical. Implementation qualifications are NOT inherited.

Preparation worktree: `/home/admin/Documents/ChatGPT/KT/kt-round8-qwen35-prep`,
branch `codex/round8-qwen35-prep`, starting commit
`d4b55412fe378a819050be704688bed38a39d5fe`.
Local SGLang gitlink `f02f304a25364d1e8fe9c52515a22f20f5089a52` is uninitialized
here and is NOT a statement of actual deployed imported code.

Model full revision resolved using `hf models info --expand sha`:
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Only config, generation config, tokenizer config/template, model index and
README were downloaded at this revision. Hashes are frozen in preparation.json.
Full tokenizer payload, license inventory and real tensor headers are pending.

Official KT main observed by read-only git ls-remote:
`f7607c0f66c643220ff43b822b24483c98c41392`.
User-supplied SGLang pin `541ddc37cbc92c60dc748db5ff1a2aad0b069a80` remains
GITLINK_VERIFICATION_PENDING. A raw qwen4_exp.py fetch at that pin returned404;
this establishes only an unresolved file path, NOT unsupported architecture.
No upstream merge/rebase or source upgrade performed.

Primary references (reviewed2026-09-19; external claims are not Ascend tests):
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/README.md
- https://github.com/kvcache-ai/ktransformers/pull/2209 — GPU/x86 kernel precedent
- https://github.com/sgl-project/sglang/pull/39126 — GB10/NVIDIA PLE precedent
- https://github.com/sgl-project/sglang/issues/38731 — roadmap, not device qualification

Use SOURCE_INSPECTED / LOCAL_VERIFIED only for corresponding real checks.
All Round9 device gates remain NOT_RUN. No weights were qualified by metadata.
