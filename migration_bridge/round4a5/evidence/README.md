# Round 4A.5 Evidence

This directory contains compact, reviewable evidence produced by the clean P2
reproduction and subsequent root-cause experiments. Large tensor dumps remain in
the A3 host artifact directory and are referenced by manifest and SHA256.

## Clean reproduction

| File | SHA256 |
|---|---|
| `p2-minimal-same-path.json` | `9d1125747de32ca2914089e3eed5501de43b767c529c082fcf755ed12ae67ce8` |
| `p2-minimal-same-path.log` | `05e29e281393c69dc74f3bb6a3d8ca216cac8f650b51b11236102c0215039f57` |

The JSON payload's internal canonical SHA256 is
`7a4e74ce97b6eab803b084b9eb6259fea33c8f9023cf30aebf80c0e80fe54ef8`.

## First matched-history capture

| File | SHA256 |
|---|---|
| `p2-v-en-01-token1-stage-comparison.json` | `8db80e2ccc25ec54170ca8f24bd2fb4c95e9bf2c03d0c3950f6428bda2f9ddb6` |
| `p2-v-en-01-per-wrapper-cpuinfer-repeats.json` | `43183d8c0bd081738381d1d56f8cc06dc3a108bf0110ea39ec1753afc03b8027` |

The matched-history capture first differs at Layer 26 `cpu_output`, while the
Layer 26 input, router values, and NPU partial are byte-identical.  The
per-wrapper CPUInfer diagnostic created four distinct WorkerPools but retained
same-path nondeterminism (8 unique outputs in 10 repeats); it is evidence
against shared CPUInfer queueing as the sole cause.

## Layer bisection (in progress)

| File | SHA256 | Result |
|---|---|---|
| `p2-bisection-l26-repeat.json` | `f9b6e6791faa710fcda61974bce19e461fdf4a7ff5ee7a5cda70a0086002df64` | exact, 1/10 hashes |
| `p2-bisection-l17-l26-repeat.json` | `1daf63623fd31c32e03da92d629b508a03ed0108bf4a0353c5f01c3bae6c6bfa` | exact, 1/10 hashes |
| `p2-bisection-l9-repeat.json` | `60572bdf01d1d615fe5b488098ea9de51f81cc27a7967215d3e5ccd0584a7b3d` | exact, 1/10 hashes |
| `p2-bisection-l9-l17-repeat.json` | `efb884c0026359cde3b99598a7ae3f7f89e1b73421a2e9ce2b06dafea886588b` | nondeterministic, 3/10 hashes |
| `p2-bisection-l1-repeat.json` | `683fe8f48be4602015010af89cb21712e9543fc1a1c66d30f29f63e231ccc0c0` | exact, 1/10 hashes |
| `p2-bisection-l1-l17-repeat.json` | `2adbfe69feddf6eeec302f70db98f3e07fbd349839707d1c82370f400a5b7e38` | nondeterministic, 5/10 hashes |

The four diagnostic placements preserve frozen P2 expert IDs.  They are not
acceptance placements.  `{1,17}` and `{9,17}` are the current minimal known
failing sets; the bisection remains incomplete.

## Parallel-execution A/B

| File | SHA256 | Result |
|---|---|---|
| `p2-l1-l17-private-output-repeat.json` | `7af3772d4150b2ef496dd7243c3cbc7ced44631467d9d5b024c01556ce687280` | nondeterministic, 2/10 hashes |
| `p2-cpuinfer1-v-en-repeat.json` | `338904b30583447851039562f226c4c6a8ec6f754b4af73e5ed82762b5de7420` | exact, 1/10 hashes |
| `p2-cpuinfer1-v-struct-repeat.json` | `79c4c33f7157ecd30f8ffe830ce77ff6af5e5f5b08691cebeb71950a1d0cf02c` | nondeterministic, 2/10 hashes |

Private LLAMAFILE/TP scratch A/B switches remained nondeterministic.  Reducing
CPUInfer to one worker fixes `v_en_01` but not `v_struct_01`; it is a diagnostic
control, not a P2 fix.

| File | SHA256 | Result |
|---|---|---|
| `p2-l1-l17-cpu1-repeat.json` | `cbe8f8772f04c96a38527a185d11789e671b4e5e2c64c2900ea75e22103880f2` | exact, 1/10 hashes |
| `p2-l1-l17-cpu2-repeat.json` | `8ce2e91abb93930bc41a097fe680c0c789f8994941a8ff121ed5d4860255b57b` | exact, 1/10 hashes |
| `p2-l1-l17-cpu4-repeat.json` | `ddaa685d81c35c7bfa538fe98bec8d09844f542baf4e8d72199ccfe2fec64239` | exact, 1/10 hashes |
| `p2-l1-l17-cpu8-repeat.json` | `6b319883197008703f437a424f33811704db6d8550a45cdc548916c859db55f6` | nondeterministic, 2/10 hashes |

## Direct CPU replay and staging isolation

| File | SHA256 | Result |
|---|---|---|
| `p2-direct-l1-l17-cpu8-summary.json` | `8ce789a27ba6c174c57decf90ca78774d18e3320877066d1b28dfe903d0777e8` | direct L1→L17 replay, 8 workers, exact for 100 repeats |
| `p2-per-layer-staging-v-en-repeat.json` | `98d221a828942ffd6ad38898b363a9dd2549ffcf26ab21c0052c7e353d3c834f` | per-layer staging, nondeterministic, 7/10 hashes |

The direct replay uses real captured CPU inputs, router IDs, and weights from
the frozen P2 GGUF.  It does not reproduce nondeterminism, while the complete
server path does; this excludes fixed-input CPU MoE computation as a sufficient
cause and rejects shared SGLang staging as a sole cause.

## Raw ACL transfer A/B

| A3 artifact | SHA256 | Result |
|---|---|---|
| `p2-input-clone/repeats-v_en_01.json` | `c1956d478c6578749959ea16f5b045135014034c24b67d15f7130380da405e76` | raw ACL D2H destination cloned; nondeterministic, 7/10 hashes |
| `p2-torch-transfers/repeats-v_en_01.json` | `76c217ecfa9c494a93306c0f842ff7f65e3d2dda4c54a94e05a5514dc230167b` | framework D2H/H2D; exact, 1/10 hashes |
| `p2-torch-transfers-struct/repeats-v_struct_01.json` | `1d312b0cf70f3f5b593caef2fe97a20eb3f7c88464dc61ecc7109eec3c661a86` | framework D2H/H2D; exact, 1/10 hashes |

The two exact runs enable only `KT_DEBUG_NPU_TORCH_TRANSFERS=1`, which swaps
raw `acl.rt.memcpy` pointer calls for blocking framework `copy_` operations.
They retain the frozen P2 model, placement, worker count, sequential control,
and request envelope.  Together with the failed D2H destination-clone control,
they attribute the blocker to raw ACL transfer integration; see
`../05_RAW_ACL_TRANSFER_ROOT_CAUSE.md`.

## Post-fix requalification

Large post-fix artifacts remain on A3 under `/workspace/results` (host-persisted
at `/home/admin/kt-artifacts/round4a5`).  Their review hashes, commands, and
gate status are recorded in `../06_P2_FIX_REQUALIFICATION.md`.  Q2 and H2
passed the unchanged numerical contract; coverage reached 4/4 layers and
16/16 selected experts.  The final quality A/B is intentionally not claimed:
the frozen 128+128 quality input is unavailable.
