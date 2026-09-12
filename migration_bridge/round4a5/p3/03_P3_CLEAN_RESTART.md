# P3 Clean-Restart Determinism

Evidence state: `A3_VERIFIED_READY`

After G1, the serving process was fully stopped and a new process loaded the
same model, tokenizer, GGUF, placement, child SHA, and runtime configuration.
Each of the nine frozen validation prompts was generated for 64 tokens.

| Check | Result |
|---|---|
| G1 process internally deterministic | PASS |
| restarted process internally deterministic | PASS |
| prompt input IDs equal | 9/9 |
| generated token IDs equal | 9/9 |
| corpus and row identities equal | PASS |

Comparison artifact: `evidence/g2-clean-restart-comparison.json`; file SHA
`c7fc2bd9cd34dc70a948627a321d2cdf52e87bdee04cc0fb0a01fbacb262cc3b`;
internal SHA
`36ff13692f8f9e205917c8699548d9f93fadbb88a350cab6ad969465b2581155`.

`P3_CLEAN_RESTART = A3_VERIFIED_READY`.
