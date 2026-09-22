# Round 6 Placement and Weight Freeze

Evidence state: `A3_VERIFIED` for S6 route capture, placement selection and D3
GGUF export/provenance.

## Selection evidence

The corrected, frozen S6 corpus was run on the All-NPU defer=0 path with radix
reuse disabled only for this offline route-capture process.  The response-side
routed-expert capturer returned the logical Top-6 IDs in rank order for the
last 64 generated positions of every prompt.

| Item | Value |
|---|---|
| S6 canonical SHA256 | `e51af126389507e03b58ae37e5b39c912ece14d54c1daa0ba4baea634ff8f9c5` |
| Prompts / positions | `12 / 768` |
| MoE layers / Top-K | `26 / 6` |
| Total logical routes | `119808` |
| Rank artifact SHA256 | `c58c9fec7ce1741f22f0c5a7fc2d1deab2e4e5452b980dfd2b864a8ab1f9f48f` |
| Route summary contract SHA256 | `3ea671e57135258c650937ffaedfc061fa57ea56171c730e5c0cf96cd45a15ad` |

Selection was performed once, before QD/HD/FD execution.  Each layer's four
CPU experts maximize total S6 route hits, then lowest (rank-6) route hits, then
use logical expert ID as a deterministic tie-break.  D3 maximizes the aggregate
score over every eight-layer contiguous window; D2 is the best four-layer
window inside D3; D1 is the best two-layer window inside D2.  This makes the
profiles nested without using qualification or held-out outcomes.

## Frozen nested profiles

| Profile | Layers | Deferred producers | Terminal | CPU experts | Placement SHA256 |
|---|---|---|---:|---:|---|
| D1 | `{7,8}` | `{7}` | 8 | 8 | `23edad71e52c05b1dc3a227ea6e27cacad0b4db6a18c4419647efae00530d39c` |
| D2 | `{6,7,8,9}` | `{6,7,8}` | 9 | 16 | `aed57f4a8d08443c1d19eceeac14b4963b5945c7fe4649079707a76667dd0e31` |
| D3 | `{5,6,7,8,9,10,11,12}` | `{5,6,7,8,9,10,11}` | 12 | 32 | `769ee8bcb2bbfe0777f184bc38407d055427a0a9cff17b51811caa8c371850de` |

The block terminal is resolved to `max_deferred=0` by the topology guard even
though the process-level requested budget remains one.

## Per-layer expert freeze

| Layer | CPU experts | S6 selected route hits | Rank-6 hits |
|---:|---|---:|---:|
| 5 | `{25,38,40,50}` | 1094 | 127 |
| 6 | `{11,37,46,56}` | 1014 | 105 |
| 7 | `{16,38,41,50}` | 1212 | 104 |
| 8 | `{8,25,33,55}` | 1026 | 96 |
| 9 | `{19,41,45,46}` | 1134 | 73 |
| 10 | `{17,44,55,59}` | 956 | 109 |
| 11 | `{16,28,43,47}` | 1135 | 105 |
| 12 | `{28,29,38,41}` | 1077 | 90 |

The canonical placement-manifest contract SHA is:

```text
5ca3db30aa401b7a5e442fd7cd4b0a617808d420d90a713e570667c101a7c5b1
```

## Frozen weight artifact

A single F32 GGUF contains all 64 experts for layers 5 through 12.  D1 and D2
consume strict layer subsets from this same artifact.

| Item | Value |
|---|---|
| Filename | `deepseek-v2-lite-round6-d3-f32.gguf` |
| Layers | `{5,6,7,8,9,10,11,12}` |
| Size | `17716742368` bytes |
| GGUF SHA256 | `4b42799ec7f143569692133aadd37f089ff0c18508411e5fd37911b62c8e323f` |
| Manifest-file SHA256 | `568c16be226c1de0ca819639e8727ee047febb1fc39f533479ff0bb22bd7c878` |
| Exporter SHA256 | `5a265bfc9a7d149ae57e5eada797e536e618ecc1939258582be884ae5ec4d09e` |
| Source config SHA256 | `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d` |
| Source index SHA256 | `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf` |
| Dtype | F32 materialization of frozen checkpoint tensors |

The manifest records all source shard hashes, every source tensor key/hash and
every exported tensor shape.  The GGUF was hashed again after the writer closed
and matched the manifest value.

## Gate

```text
R6_G4_PLACEMENTS = FROZEN
R6_G4_GGUF = FROZEN
PLACEMENT_RESELECTION = PROHIBITED
```
