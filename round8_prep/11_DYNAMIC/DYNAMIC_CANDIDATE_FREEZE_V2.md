# Repaired Dynamic candidate v2 — entry decision

## Closed entry gates

- Real model warning attributed to transposed NZ expert slice copy, not audit.
- Actual old function fails the no-warning operator regression; new function
  passes three identity changes against independent CPU weight tensors.
- Fixed full model: 120 updates, 40 layers x3, exact copied-weight/mapping/mask
  assertions; workload A repeated selection exact; workload B changes all layers.
- Five pre/post-fix DEV token trajectories exact; fixed server warning count zero.
- New process with no forward hooks: same five DEV trajectories exact to the
  audited fixed process; independent raw-response audit 5/5 PASS. No offset warning.
- Static 2304-token stability/restart evidence retained with cgroup/HBM plateau
  and explicit limits on what the memory collector measured.

The hook-free restart does not independently record selected-expert IDs. Its
token parity is not presented as an independently observed selection trace.
Audited within-process selection/generation evidence is retained separately.

## Selection and scope

Freeze threshold256 with existing chunk512. This is a DEV-compatible engineering
candidate that exercises Dynamic on the authorized medium256 workload, not a
claim of threshold optimality. DEV256/512's original triggers both used512 rows;
no comparative performance claim is inferred from that matrix. DEV1024 is
explicitly skipped by the user, with its failed no-trigger result preserved.

Candidate SHA256:
`69892f31dcbb450105a4adf153622c7bf6f5823292bfb8038ac9ae60160e55b9`.
Configurations/tool hashes reside in `qwen-dynamic-candidate-v2/`.
The commit containing these bytes must precede uploaded freeze-ack and any
held-out/CMMLU request. No held-out output has been used for this decision.

## Ordered execution after commit

1. Held-out128 on lane-a/NPU4/CPU0–63, with established generation/weight auditor.
2. CMMLU256 concurrently on lane-b/NPU6/CPU64–127, no model hooks; same manifest,
   tokenizer, one-token candidate-logprob scoring, reference222/256, minimum210.
3. Independently audit both raw outputs. On PASS, release same-machine isolated
   hook-free Static/Dynamic ABBA on prep/NPU2/CPU128–191: short64/32, medium256/64,
   three blocks, no long request. Never measure while quality lanes are active.

Quality records actual per-request layer-update counts from existing production
logs; these non-timed observations are not throughput evidence. Performance
reports observed E2E latency; it must not mislabel E2E-derived rates as isolated
decode throughput or claim separately measured TTFT.

All statuses beyond entry remain NOT_RUN until real artifacts establish them.
Dynamic is not yet quality-qualified or recommended as default.
