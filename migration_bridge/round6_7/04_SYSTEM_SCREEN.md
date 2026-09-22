# All-MoE system screening

Current state: execution in progress; no all-MoE Hybrid profile is qualified yet.

Gate evidence per profile is emitted under the active study directory:

| Gate | Artifact | Acceptance |
|---|---|---|
| Same-process/prefix | compact/requests.jsonl | 32 requests, exact prefixes and repeats |
| Clean restart | restart/requests.jsonl | Four16-token trajectories equal prior process |
| Observer parity | observer/requests.jsonl | Four8-token prefixes equal unobserved run |
| Physical ownership | observer/physical-structure.jsonl | 26layers, frozen IDs/counts/dtypes |
| CPU/NPU routing | observer/routing.jsonl | Finite Top6, all26layers, each Hybrid layer CPU hit |
| Local operator safety | operators/L01..L26.json | 52expert instances ×3actual decode operands; relL2<=.01 |
| Repeat operator output | same operator artifacts | Three repeats, one byte hash per operand |
| CPU-not-hit control | same operator artifacts | Real operand, controlled NPU-owned route, CPU partial exact zero |
| Lifecycle | each session/lifecycle.json | Owned descendants fully reaped before next process |
| Provenance | STUDY_FREEZE.json | Exact source/asset/profile/tool hashes, checked between phases |

R64 has no CPU experts; its CPU-only local operator/control gate is explicitly
N/A. Controlled no-hit samples do not claim to be naturally occurring all-NPU
routes. Existing weight identity and production semantic audits are retained;
this compact screen samples integration behavior, not exhaustive certification.

The deployment is Ascend910B3 on110.129.0.22. Historical A3 qualification labels
are not used for new executions on this server. Startup/tool failures retain
their own versioned evidence and do not count as quality answers.
