# Git Commits

## Base

- parent base: `c2a456aec16846d353bd3075361d2cda6a3e085c`
- parent branch: `feature/kt-round4a3-numerical-acceptance`
- SGLang base/final: `8a4bb4c850860377d9880b8c0fa570ca2d0d19e1`
- SGLang branch: `feature/kt-ep-round4a3-numerical-acceptance`

## Diff discipline

Round 4A.3 adds only `migration_bridge/round4a3` tools, frozen corpora,
evidence, and documentation. Production KTransformers and SGLang code are
unchanged. Final parent commit SHA is recorded after the documentation commit.

## Commits

- `6453a54` `test(round4a3): add numerical qualification harness and corpora`
- `1a01f19` `test(round4a3): record Q and heldout qualification evidence`
- final documentation commit: the commit containing this file; its full SHA is
  reported in the handoff because a commit cannot embed its own hash
