# First-Divergence Attribution

## Cases: `v_struct_03` at 16, 32 and 64 tokens

Prompt: `Write pseudocode to test whether an integer is even.`

The 16/32/64 validation cases share the same first mismatch at generated token
index 9:

- common history: `[185,185,62031,484,64645,25,185,185,12]`
- all-NPU next token: `8828`
- Hybrid next token: `1273`

These are three failed Corpus V requests. Because all three lengths use the
same prompt and deterministic common prefix, one shared capture represents
their identical first-divergence state. The histories are identical through
index 8, making this a teacher-forced same-history comparison at the first
divergence; later free-running tokens are excluded from causal attribution.

## Logit-margin evidence

The API exposes quantized logprobs rather than raw full logits. For the two
decision candidates:

| Candidate | all-NPU logprob | Hybrid logprob |
|---|---:|---:|
| 8828 | -2.521951675415039 | -2.5539395809173584 |
| 1273 | -2.646951675415039 | -2.5539395809173584 |

Thus the all-NPU margin `8828 - 1273` is `+0.125`, while the Hybrid margin is
`0.0`; delta margin is `-0.125`. Hybrid also ties token 17570 at the same
reported value. Greedy tie-breaking selects the lower token ID 1273.

## Pass-level localization

For prefill plus ten decode passes, Layer 17 input, TopK IDs and TopK weights
are bitwise equal between all-NPU and Hybrid for passes 0-9. Pass10 is the first
different input and is downstream of the token flip.

Selected CPU routes before the flip include:

- pass0: E36 in prefill;
- pass1: E6 at TopK position 1;
- pass2: E25 at position 0;
- pass6/pass7: E25 at position 0 and E6 at position 5;
- pass8: E25 at position 1 and E36 at position 5;
- pass9: E25 at position 0.

Single-route comparison against captured NPU down values showed sparse BF16
differences. Examples:

- pass1 E6: 28 BF16 term elements differ, max abs `8.249282836914062e-05`;
- pass8 E25: one BF16 term element differs, max abs `0.00072479248046875`;
- pass8 E36: 95 BF16 term elements differ, max abs `0.0002899169921875`;
- pass7 E25 and E6: both route terms are bitwise equal;
- pass9 E25: bitwise equal.

This input-dependent sparsity identifies E36 as the largest element-count
contributor in the immediately preceding non-exact pass, while proving that no
expert is always wrong. The complete 15-subset campaign further shows that E6
or E25 alone can trigger the token flip, E36 alone does not, and E36 can cancel
E25 at the final decision. There is therefore no globally dominant expert;
dominance depends on hidden state, error direction and candidate margin.

Compact responses are stored in:

- `evidence/round4a1-struct03-allnpu-response.json`
- `evidence/round4a1-struct03-hybrid-output-response.json`
