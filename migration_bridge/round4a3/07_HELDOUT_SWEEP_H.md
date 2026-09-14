# Held-Out Sweep H

Evidence state: **A3_VERIFIED**.

- prompts: 12
- positions per prompt: 64
- total positions per mode: 768
- All-NPU manifest SHA: `b3ca6858631c0fdd0b19f786f87bc627c4aa39b3d9a003f9bf0220e7a5ceee43`
- Hybrid manifest: `evidence/manifests/h-hybrid.json`
- both sweep exit codes: 0
- all full logits finite: yes
- sampling-pass consistency: pass

The same frozen All-NPU histories were used for both modes. H was never used to
modify epsilon, C, or the tie-set rule.
