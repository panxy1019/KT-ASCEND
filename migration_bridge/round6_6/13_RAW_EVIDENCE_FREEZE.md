# Closed V1 raw evidence content freeze

After both model processes stopped and the failed quality sequence was reaped,
the read-only content manifest covered these closed target directories:

- `/workspace/results/r65-r66-qd6-v1`
- `/workspace/results/r65-r66-hd6-v1`
- `/workspace/results/r65-r66-final-screen-v1`
- `/workspace/results/r65-r66-hd6-outlier-operators-v1`

Files:1,076,644. Total bytes:9,238,283,445.
Manifest:`/workspace/results/r65-r66-raw-evidence-v1.jsonl`.
SHA256:`c720fda42d3f47467fae1bb1f5cf51bb9c4e62f70ae44cdae75ab5a5a7174fb3`.
Each entry records path,byte size and SHA256; symlinks or files changing during
hashing are rejected. The guard exited zero and reaped its owned descendants.

The multi-million-line raw manifest and tensor files remain in target artifact
storage; they are not represented as if all raw tensors were committed to Git.
Compact audits,raw quality requests,results and lifecycle evidence are committed.
Earlier B0 raw per-case artifact hashes and weight provenance remain separately
bound in the B0 evidence. No model/GGUF bytes were rewritten by this operation.

This freeze preserves the failed V1 result. It is not a passing-release tag or
a waiver of the quality failure.
