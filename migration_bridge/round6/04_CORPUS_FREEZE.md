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
| Tokenizer class | `LlamaTokenizer` |
| Special tokens added | `NO` |

## Prompt corpora

| Corpus | Prompts | Positions target | Purpose | Canonical payload SHA256 |
|---|---:|---:|---|---|
| S6 | 12 | 768 | defer=0 route stress and one-time placement selection | `a7635497ab7424e7b0772c65c668751407b84d25719ff39b6469758bdab02cee` |
| QD | 16 | 1024 | Deferred numerical contract derivation | `1cbd5d9642c5a4e9ac3781d56b79d3a92196b30da2ac7ba0ebb36b7d7cef1a25` |
| HD | 16 | 1024 | independent Deferred held-out validation | `666d5b0ab526023785bdc05f585f6b08a84900659c1d7bda09465feb2b601c19` |
| FD | 8 | 512 | independent free-generation qualification | `25a65097d06c4fa17d853e6aa73acecd98f83b4ec56490ac2dca50cff9c5fd58` |

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
uniqueness and prior-corpus exclusion before writing canonical JSON.  Repeating
the tool in the A3 disposable container produced identical embedded hashes.

## Gate

```text
R6_G3_PROMPT_CORPORA = FROZEN
BASELINE_HISTORIES = PENDING_ALL_NPU_CAPTURE
```
