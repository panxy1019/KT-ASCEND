# Stability and Memory

Status: `NOT_RUN_P2_BLOCKED`

The planned P3 campaign of 2,304 generated tokens was not authorized because
P2 failed same-path determinism. No P3 crash, deadlock, RSS, or HBM conclusion
is claimed.

The earlier P1 controlled lifecycle did complete 1,000 overlap forwards with a
single routed-output hash. That P1 result cannot substitute for the required P3
campaign.

P2 itself showed no process crash, deadlock, traceback, or non-finite pairwise
output during the qualification subsets. Its blocker is more fundamental:
identical repeated greedy requests produced different token sequences in both
overlap and sequential-control modes. Memory qualification was therefore
stopped before a long campaign could create a misleading partial PASS.
