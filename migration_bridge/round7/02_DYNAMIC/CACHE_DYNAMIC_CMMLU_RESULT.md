# Frozen1024 Dynamic-enabled CMMLU result

2026-09-18 A3 execution. Runtime-v8, originalNPU2/CPU128–191/C64,
tools-v25, committed threshold1024. Static/Dynamic HOLDOUT concurrently running
on separate owned devices: elapsed times below are descriptive, NOT performance.

CMMLU256:174 correct/256,67.96875%, invalid0. FrozenR64 reference174/256;
observed delta0pp. Product -5pp guardrail PASS (minimum162 correct).

**Actual trigger rate0/256=0%; migration/update calls0.** This is short-request
Dynamic-enabled safety, NOT triggered migration task-quality certification.
Long triggered system probes independently passed; Dynamic held-out and formal
long-generation performance qualification are still pending. Do not hide this
coverage limitation in final ADR or promote a generic Dynamic READY from CMMLU.

Local raw audit:256 unique IDs, raw correct sum174 equals summary, invalid0,
complete lifecycle and owned descendant reaping. Request-time sum682.655579s,
full lifecycle841.464042s (about14.02min). No isolated-speedup claim.

Archive `raw/round7-cache-dynamic-quality-v1.tar.gz` SHA256:
`279c1126b682f77152f5f22bbcb1aec037289c6cf353e30848872c496c82bc45`.
SCP completed, local SHA matched host, raw audit PASS. Includes config/protocol,
explicit source-bound CPU operator reuse, raw responses, telemetry and lifecycle.

StaticHD and DynamicHD remain running. Quality cannot override a held-out
system failure, and no formal performance may run until both reaped.
