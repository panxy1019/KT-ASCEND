# Round 4A Final Commits and Tags

## Immutable anchors

| Purpose | Ref / SHA |
|---|---|
| Round 4A.5 friend takeover | `round4a5-friend-takeover-305b843` → `305b84359d6a646ab260a16d004d19147103d561` |
| framework-managed transfer fix | `00c7acfe3569a6a7a6af5b9891e9989e334dffb3` |
| P2 final freeze | `round4a5-p2-a3-verified` → `d4830525c6806d5d587b3096634614aad7fef181` |
| P3 execution documentation base | `25821cd22d6a1c63f43467a9f0e07bf5958182b3` |
| Round 4A final freeze | `round4a-tp1-multiplacement-a3-verified` → immutable annotated-tag target |

## P3 commit discipline

P3 introduced no production implementation or SGLang child change. Changes
after the P2 tag are evidence and documentation only. The final annotated tag
is created on a clean tree after:

1. A3 evidence collection and byte-hash inventory;
2. P3 gate documentation;
3. Round 4A final gate/provenance documentation;
4. local consistency checks.

The exact target is authoritative via:

```bash
git rev-parse round4a-tp1-multiplacement-a3-verified^{}
```

The tag must not be moved. Round 5 must branch from it and maintain an
independent evidence chain.
