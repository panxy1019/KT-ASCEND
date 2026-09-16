# Test-process cleanup supplement

2026-09-16 read-only container inventory observed1774 zombie processes. Many
have historical low PIDs and PPID1. The container's long-lived PID1 does not
reap these exited children. No historical process was signalled or removed;
the healthy batch campaign remains on its original tooling.

The retained guard already signals only the session it creates and waits for
its direct launcher. A launcher can exit before its multiprocessing workers,
leaving orphaned exited descendants for PID1. This is test-process cleanup,
not evidence of a Deferred tensor-lifetime or numerical failure. Zombie
entries do not retain model HBM; they still consume process-table entries,
so long-run lifecycle reporting must not simply ignore the issue.

Future versioned tooling wraps the unchanged group guard with Linux
PR_SET_CHILD_SUBREAPER. It only adopts orphaned descendants of this new guard,
then waitpid-reaps them after the existing guard has completed its cleanup.
No additional signals are sent. It cannot adopt or reap pre-existing PID1
children or unrelated jobs. A timeout reports remaining descendants rather
than silently claiming successful cleanup.

`guard/reaping.json` records exact reaped PIDs/wait statuses. Source hash is
added to run provenance. Full local tool suite:53PASS, including a real
fork/grandchild orphan-reaping test. A3 validation is still pending a new
owned test run; this does not retroactively change prior guard evidence.

No production source, worker topology, stream ordering, host library, or
container PID1 configuration changed. Prior zombie entries remain an
environment limitation; no zero-process-leak claim is made for historical runs.
