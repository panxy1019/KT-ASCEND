# Round 6 Corpus Freeze

Evidence state: `A3_VERIFIED` for tokenization and prompt-corpus hashes.

The four Round 6 corpora were frozen before route profiling, Deferred placement
selection, or any defer=1 model output was observed.  They are mutually
disjoint and were checked against the frozen Round 4A S/V/T, Round 4A.3 Q/H,
and Round 4A.4 Q2/H2/F prompt texts.

## Frozen tokenizer

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V2-Lite` |
| Revision | `604d5664dddd88a0433dbae533b7fe9472482de0` |
| `tokenizer.json` SHA256 | `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669` |
| Tokenizer class | native `tokenizers.Tokenizer` loaded directly from the frozen JSON |
| Special tokens added | `NO` |

## Prompt corpora

| Corpus | Prompts | Positions target | Purpose | Canonical payload SHA256 |
|---|---:|---:|---|---|
| S6 | 12 | 768 | defer=0 route stress and one-time placement selection | `e51af126389507e03b58ae37e5b39c912ece14d54c1daa0ba4baea634ff8f9c5` |
| QD | 16 | 1024 | Deferred numerical contract derivation | `fe28d7fe4b4421b53901950b41a7d98b58650de359bd5141e9d05ef71e6bf6f1` |
| HD | 16 | 1024 | independent Deferred held-out validation | `0ee3b50c9d02515f1492463ea1af5d507ad72267d43d5e0962e6bf9c6619adba` |
| FD | 8 | 512 | independent free-generation qualification | `197f8a9a4bca0349a2984d603e71c1722cea1828fdbff30be00fe7441a4aa1a6` |

Each category family is represented in every corpus: English factual, Chinese
factual, math/numeric, structured JSON, and code/reasoning.  QD and HD each
contain 16 prompts; S6 and FD are used only for their declared purpose.

## Protocol

```text
teacher-forced positions = 64
candidate top K = 32
temperature = 0
seed = 0
```

The prompt text and tokenizer input IDs are immutable at this gate.  All-NPU
baseline histories have not yet been generated.  Their future evidence
manifests must reference the corresponding canonical prompt-corpus SHA above;
capturing those histories does not alter these frozen files.

The existing Round 4A quality corpus remains frozen independently:

```text
c7bcb09c72f4a7213d0ecdb080f8e88e983ab1a491d4304f448bce28d8f7e1ce
```

It is not used for placement or numerical-bound fitting.

## Reproduction

`tools/freeze_round6_corpora.py` validates the tokenizer hash, cross-corpus
uniqueness, prior-corpus exclusion and non-empty tokenization before writing
canonical JSON.  A rejected first attempt used the Transformers 5.3 slow
`LlamaTokenizer`, which produced empty IDs for pure Chinese prompts.  No
placement had been selected and no Deferred output had been observed.  The
corrected implementation loads the immutable `tokenizer.json` natively; it was
verified against the already-frozen Round 4A English and Chinese token-ID
vectors before these final hashes were produced.

## Gate

```text
R6_G3_PROMPT_CORPORA = FROZEN
BASELINE_HISTORIES = PENDING_ALL_NPU_CAPTURE
```
