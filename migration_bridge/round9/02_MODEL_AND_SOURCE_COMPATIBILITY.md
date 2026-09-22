# Model/source compatibility — preliminary source inspection

Frozen config is stronger evidence than model-name similarity, but weaker than
checkpoint headers and execution. See evidence/model_metadata and manifest.

| Component | Observed metadata | Required before full-model entry |
|---|---|---|
| Registration | Qwen4ExpForConditionalGeneration; qwen4_exp; text qwen4_exp_text | Locate actual KT/fork/upstream implementation and Ascend dispatch |
| Main model | 48layers,hidden2560,512experts,Top10,expert/shared intermediate640 | Real tensor census, source dtype, shared/scaling ownership |
| Attention | 36linear +12full, repeat3linear/1full | GDN reset/chunk state and original QSA indexer/selection/causality |
| GR | hc_count4,hc_lowrank320 | Actual gate/norm/residual formulas and lifetimes |
| PLE | config ple_layer_ids=[2];128 index shard keys under language_model.layers.1.ple | Resolve numbering from code; exact hash/history/row layout and sizes |
| Expert weights | 96packed main keys: gate_up_proj and down_proj for48layers | Packing axes, transpose, per-expert slicing, ND/NZ roundtrip |
| Auxiliary | MTP config1layer,vision config present,language_model_only=false | Explicit loader partition, no MTP/vision conversion mixed into expert bank |
| Tokenizer | tokenizer_config includes chat_template | Full tokenizer hash, special IDs, nonthinking/greedy protocol freeze |
| CPU | New intermediate640 | Real320+320 alignment, F32 merge and Top10 shape tests; no inherited PASS |

Index contains1658 keys across131 shard files; metadata.total_size is
359999963128 bytes. It does not provide tensor shapes or prove byte identity.
Configuration dtype is bfloat16; verify checkpoint before calling it BF16 source.
QSA indexer_budget2048/compress_ratio4 does not mean total KV retention2048.
Max config context262144 does not qualify any Ascend context length.

Model converter cannot reuse Qwen3.5 names/axes without inspection. PLE [2]
versus indexed layer1 is a source-audit question, not an automatic off-by-one fix.
Unknowns: actual supported fork file locations, QSA NPU primitives, GR equations,
PLE row shapes/ID algorithm, source packing/dtypes, operator ABI compatibility.
