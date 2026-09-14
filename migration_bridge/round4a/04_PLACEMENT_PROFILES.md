# Placement Profiles

Status: `PASS`.

The deterministic builder emits JSON and SGLang `logical_count` PT files for:

- P0: Layer 17 / Expert 8.
- P1: Layer 17, four CPU experts, retaining Expert 8.
- P2: four depth-distributed MoE layers, four CPU experts per layer, including
  Layer 17 / Expert 8.
- P3: eight depth-distributed MoE layers, four CPU experts per layer, including
  Layer 17 / Expert 8.

Every emitted profile is independently validated for mask shape/dtype,
CPU/NPU partition completeness, exact selected counts and physical NPU counts.

Frozen placements:

```text
P0  L17:{8}
    SHA256 f3e14a883aa7910c264d77cc927c4a27d9127c91ab8f786563ebe1ffcaf2d122

P1  L17:{6,8,25,36}
    SHA256 9548bf6e06014e034c6a6650af3a891a546f70d8c8d79b059174ba571c44471f

P2  L1:{31,43,50,57} L9:{38,41,45,46}
    L17:{6,8,25,36} L26:{10,26,30,56}
    SHA256 f6d4e9c6a2e5e8060e846dbc7c628d069c9aa6150aaa1e2690ea28a51ba286a3

P3  L1:{31,43,50,57} L5:{14,38,39,41}
    L8:{1,17,33,45} L12:{35,41,46,58}
    L17:{6,8,25,36} L19:{14,32,47,62}
    L22:{21,60,61,62} L26:{10,26,30,56}
    SHA256 77f7d96e0b180242264f1d6eec3e4d9d16d158e9e0f9a51eac51af88e829391d
```

P1/P2/P3 selected layers have 60 physical NPU experts. Every unselected MoE
layer has 64. All dense layers derive an all-accelerator bool mask and skip the
KT wrapper.
