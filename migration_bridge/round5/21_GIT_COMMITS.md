# Round 5 — Git Commits and Freeze

## Immutable references

| Reference | Object |
|---|---|
| Round 4A annotated tag | `round4a-tp1-multiplacement-a3-verified` |
| Round 4A tag object | `9e2c9e220337f57b55f1fe9b50da0862ccc0cf64` |
| Round 4A target | `096a6ef860a01dc0a3c8f2cc5618a396425ee09b` |
| pinned upstream parent | `440df948027128318bc30a11a545433eb0c7d383` |
| pinned upstream SGLang | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` |

## Aligned parent patch stack

| Commit | Subject |
|---|---|
| `dfcebfb` | guard SGLang CUDA-only capability probe |
| `bbf8b36` | support DSV2-Lite LLAMAFILE tail dimensions |
| `255cf14` | preserve validated Hybrid numerical semantics |
| `6368cde` | close CPUInfer callback state lifetimes |
| `0d8fb1d` | preserve DSV2 expert mapping contracts |
| `c84537b` | synchronize the active accelerator device |
| `bf1628b` | preserve NPU router FP32 fallback |
| `e07051f` | bridge CPUInfer to active accelerator streams |
| `e98916c` | preserve native ARM kernel identity |
| `99b3906` | preserve standalone CPU expert execution |
| `9743248` | preserve LLAMAFILE expert mapping |
| `01c99b9` | expose pinned transfer contract |
| `958226e` | add Ascend framework transfer regression |
| `be5b486` | close callback worker lifecycle |

Final aligned parent:

```text
be5b48621db32fb894402e3e886060dea263cdc9
```

## Aligned SGLang patch stack

| Commit | Subject |
|---|---|
| `89288d20e` | guard CUDA capability probe during import |
| `44a72c677` | preserve DSV2 expert mapping contracts |
| `ef0b59a92` | preserve FP32 Hybrid expert contributions |
| `b4f9cb959` | synchronize the active accelerator device |
| `c90482bcc` | avoid CUDA JIT for router linear |
| `412219b4b` | use active accelerator stream in KTEP |

Final SGLang child:

```text
412219b4b05a888808f0b88d2963a8b6164a0fda
```

## Final tag

```text
name: round5-upstream-aligned-tp1-a3-verified
annotated tag object: 94baf6272dd86eae7e465a04f4ead630032400d4
dereferenced target: be5b48621db32fb894402e3e886060dea263cdc9
```

The tag points at the clean upstream-based runtime source.  Round 5 reports,
tools and compact evidence remain on the separate
`feature/kt-round5-alignment-audit` branch so documentation does not alter the
qualified runtime target.

The audit-side qualification freeze is commit
`38b9ab1b5` (`docs(round5): freeze upstream-aligned TP1 qualification`), with
compact evidence archived immediately before it at `5b3eaf2b4`.
