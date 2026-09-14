# D1 Same-Process Qualification Progress

Evidence state: `A3_VERIFIED` for the completed b681 build. The later
target-row numerical instrumentation build is undergoing its own full rerun.

## Completed b681 matrix

On parent `b6815196d8bfd531f810ba34372999dc77911b9c`, SGLang child
`eaa58ccea4ee372b8b207dd16f398eafe3da37ff`, D1 placement SHA256
`23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c`
and Round 6 GGUF SHA256
`4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f`,
the default production-overlap service completed:

```text
S6 prompts: 12/12
same-process repeats: 10 per prompt
generation: 64 tokens
independent prefixes: 8, 16, 32 tokens
repeat exact: 12/12
prefix exact: 12/12
trace/debug instrumentation: OFF
```

The runner's canonical payload SHA256 is
`ab66704c4f84b67dde22a1c2e9d4482f6fb72a2c9587e7fb452fe499b0727913`;
the raw result JSON SHA256 is
`cc6389e25a1e7d924b2b9508dfc001a0e9b04e64f6c648d8280cce4b52b3d831`.
All 12 reference trajectories were also byte-for-byte token-identical to
the earlier pre-final pending-clear matrix. The comparison reported
`all_exact=true`.

Raw A3 artifacts remain at
`/home/admin/kt-artifacts/round6-deferred/d1-determinism-final-a/`.
Portable copies of `result.json`, `compare_pre_final.json` and the A3 hash
manifest are in `evidence/d1-final/determinism-b681/`.

## Exact-code rerun

Parent `e3084d309d6f39b8a9f7185d36ac27e4b77526d7` adds only opt-in
target-row numerical telemetry and evidence tools. Its installed wheel passed
7/7 parent selection and 7/7 SGLang child alignment tests. Nevertheless,
the full same-process matrix was repeated on that exact build rather than
promoting the earlier wheel's result across commits. It again passed all
12/12 prompts, with 10/10 unique trajectories per prompt and exact 8/16/32
prefixes of each 64-token reference. The 12 reference trajectories were also
all exact against the b681 build.

```text
e308 result JSON SHA256:
171b717bfec144fec3eaa37acd7baf090ab19d5d83271fc1bcde7b23c9639edc
e308 canonical payload SHA256:
5daceab68b6378091105cea755d97b535b16c938f93da57b44b40a3e6f1b1290
comparison b681 vs e308: all_exact=true
```

A portable copy of the e308 result JSON, comparison and hash manifest is in
`evidence/d1-final/determinism-e308/`. The full server log remains in A3
`/home/admin/kt-artifacts/round6-deferred/d1-determinism-e308-a/`.

```text
D1_SAME_PROCESS_E308 = A3_VERIFIED
D1_CLEAN_RESTART_E308 = IN_PROGRESS
D1_FULL_MODEL_NUMERICAL = NOT_RUN
D1_DOWNSTREAM_QUALITY = NOT_RUN
```
