# Round 2C Git Commits

Base: `6f50c3782f6002940dedcbbc74c6af980fc0d862` (`round2b-a3-verified`).

1. `f6bd4a6` feat(moe): add fixed hybrid expert placement contract
2. `150d99a` feat(moe): add sequential single-layer CPU NPU hybrid path
3. `686f203` test(ascend): validate deterministic sequential hybrid experts
4. `0881b06` test(moe): cover hybrid routing and expert mappings
5. `ca08452` test(moe): validate overlapped hybrid correctness
6. `0493d6d` test(moe): validate hybrid lifecycle and stress
7. `b8ba787` fix(moe): synchronize overlapped host transfer before merge
8. documentation commit containing `migration_bridge/round2c/`

The production synchronization fix is isolated in commit 7. Raw A3 logs and build products are retained remotely and are not committed. Pre-existing untracked Round 1/1.5 material under `migration_bridge/` was not staged or modified.
