# Static mask and measurement freeze

Status: deterministic profiles generated; runtime gates pending. No frequency
statistics or quality outcome was used to choose uniform experts.

MoE layers1–26,64experts per layer. R8⊂R16⊂R32⊂R64 resident IDs:
`j*(64/R)` for `j=0..R-1`. Same per-layer budget; other layers remainR64.
Source format is the retained `logical_count` tensor[1,27,64],int64; only
resident positions1, CPU positions0, dense-layer frequency0. Global requested
resident ratio is derived from exact total NPU experts, not simplyR/64 for
single/four-layer profiles. Actual generated mask must be checked at runtime.

Local builder6testsPASS. Target builder/pairwise suite12testsPASS. Parent
repository's build* ignore rule required explicit tracking of build_ratio_profiles.py.

Frozen compact numeric fixture: fixtures/numerical.json, first4Q2 and first4H2
in existing corpus order,64positions each. B_pair2.1875,K32; no fitting.
Natural R64 logits generated on the same target/runtime; heavy candidate consumes
exact baseline history via existing default-OFF teacher-forcing instrumentation.
Forward captures validateRID/sequence/input/position/KVlength/logit SHA and account
for exactlyone production-overlap lookahead per64-target request. Any extra or
incomplete forward stops comparison. This is compact requalification, not new
held-out threshold derivation.

Scheduling texts frozen before frequency observations in fixtures/scheduling_texts.json.
Each split16prompts,4percategory. Tokenization/duplicate audit must complete before
DEV routing fit; HOLDOUT cannot modify selection rules. Formal quality remains
separate and uses unchanged Round6.4 fast corpus only for final Deferred relevance.

Static timing workload will use retained192inputIDs and a deterministic1024token
extension,64generated tokens each,2warmups/3timed. CacheOFF,GraphOFF,batch1,
same4096KV token budget and2048context/prefill cap. Diagnostic capture disabled
in timed phase. One frozen CPU-worker setting across the Pareto sweep.

Full newpool SHA20114dba7790a9b6be40a17ffb3881c516b81ea86b8f2fcfd3ea61d052e96fe2;
all4992projections BF16 byte-identity verified. It is a new artifact, not the
historical D1/P3 GGUF relabeled as full-layer data.
