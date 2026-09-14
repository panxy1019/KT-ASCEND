# Prompt Corpora

Round 4A uses three disjoint, frozen corpora:

- Corpus S: 12 prompts used only for all-NPU route profiling and placement.
- Corpus V: 9 held-out prompts used only for A/B correctness.
- Corpus T: 6 further prompts used only for stability campaigns.

`tools/freeze_prompt_corpora.py` writes the exact prompt text, tokenizer input
IDs and canonical SHA256 for each corpus. Placement may only depend on Corpus S.
Corpus V and T must never be used to reselect experts.

Frozen hashes using the DeepSeek-V2-Lite revision `604d5664` tokenizer:

```text
Corpus S: count=12 sha256=6f743d4d8caa5f7480eee9dc03d4e3af4f647cdf255e1abd49ba18fc63bb10d2
Corpus V: count=9  sha256=9ae547d3fef84f097b71eb944952e708298168be2083d1b1ce4faff76d03268e
Corpus T: count=6  sha256=e82fe338c2785b0866dff7cd5a85f236620c72558ec758637fe8bbbcc595e700
```
