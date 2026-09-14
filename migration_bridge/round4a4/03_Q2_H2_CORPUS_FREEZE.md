# Q2 / H2 Corpus Freeze

Evidence state: **CODE_INSPECTED** and tokenizer freeze **A3_VERIFIED**.

| Corpus | Prompts | Positions target | Internal SHA256 |
|---|---:|---:|---|
| Q2 | 18 | 1152 | `551fc4fdf75053eda18d511ab2b479907576bd31989460281ef2424689dd3e07` |
| H2 | 16 | 1024 | `b1874527eca8c01ab3df16215679eb7365a938b2c89dcb910334ad70bba7b787` |
| F | 6 | free-generation only | `627d1feaebefeae6c116e5112dcf9e4177265e2a8567c6245dea790aa0b031c3` |

Q2 contains the nine frozen Q prompts plus nine new qualification prompts.
Its normalized categories are English factual 4, Chinese factual 4,
math/numeric 3, structured JSON 3 and code/reasoning 4.

H2 is wholly new: English factual 4, Chinese factual 4, math/numeric 3,
structured JSON 2 and code/reasoning 3. F is disjoint from Q2 and H2.
The freeze tool rejects overlap with old H, selection S and stability T. Old Q
overlap is allowed only for the explicitly reused nine Q rows inside Q2.

All input IDs were generated on A3 with the frozen local
DeepSeek-V2-Lite-604d5664 tokenizer. Protocol is temperature 0, seed 0,
64 teacher-forced positions and candidate top-K 32. Baseline token histories
are frozen as separate All-NPU evidence before the corresponding sweep; they
are not fitted numerical parameters.

Quality corpus D is frozen separately before WP12 and may not be used in Q2 or
H2 derivation.
