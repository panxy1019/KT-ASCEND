# Round 2B Git Commits

Base: `540ccbc28b1fa9327b90dc86c244e0df707409d8` (`round2a-a3-verified`).

1. `d8da19a` refactor(cpuinfer): add device-neutral stream callback abstraction
2. `495b10c` feat(cpuinfer): add Ascend runtime vendor adapter
3. `2087b98` feat(ascend): expose native stream handle to kt-kernel
4. `6ed3b4f` test(ascend): validate CANN host callback ordering
5. `b120c84` test(cpuinfer): validate CPUInfer on Ascend stream callbacks
6. `54bc1ec` test(ascend): validate pinned host transfers
7. `d2d8be0` fix(cpuinfer): release device callback state safely
8. `d07604c` test(ascend): validate end-to-end runtime pipeline
9. documentation commit containing `migration_bridge/round2b/`

The production lifetime defect was intentionally isolated in commit 7. Raw A3 logs and build products are retained remotely and are not committed. Pre-existing untracked `migration_bridge` material from Rounds 1/1.5 was not staged or modified.
