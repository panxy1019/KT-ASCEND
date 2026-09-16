"""Read-only A3 host monitor of one container's exact owned guard descendants.

No signals, library installs, device allocations, or workload modification.
Queries NPU0 only. Emits JSONL to stdout for project artifact capture.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

CONTAINER = "b91b94ef502933c63247d03ae6150e3c72ce29061c4034d2c8b435b18b6d8a1f"


def descendants(records, roots):
    found = set(roots)
    while True:
        expanded = found | {pid for pid, (parent, _) in records.items() if parent in found}
        if expanded == found:
            return found
        found = expanded


def proc_memory(text):
    return {int(pid):int(mb) for pid,mb in re.findall(
        r"Process id:\s*(\d+).*?Process memory\(MB\):\s*(\d+)", text)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--duration-seconds", type=int, required=True)
    p.add_argument("--interval-seconds", type=int, default=5)
    p.add_argument("--descendant-runs", action="store_true", help="Observe sequential child runs under this exact project artifact root")
    args = p.parse_args()
    if not args.run_root.startswith("/workspace/results/r64-") or not 1 <= args.interval_seconds <= 60 or not 1 <= args.duration_seconds <= 172800:
        raise ValueError("MONITOR_SCOPE")
    def read(command):
        return subprocess.check_output(command, text=True, timeout=15)
    identity = read(["docker", "inspect", "--format", "{{.Id}}", "kt-r6-deferred-ascend"]).strip()
    assert identity == CONTAINER, "CONTAINER_ID_DRIFT"
    end = time.monotonic()+args.duration_seconds
    while time.monotonic() < end:
        item = dict(time_unix=time.time(), run_root=args.run_root, npu=0)
        try:
            output = read(["docker", "top", CONTAINER, "-eo", "pid,ppid,args"])
            records = {}
            for line in output.splitlines()[1:]:
                pid, parent, command = line.split(None, 2)
                records[int(pid)] = (int(parent), command)
            suffix = r"/[^ ]+/guard" if args.descendant_runs else r"/guard"
            marker = re.compile(r"run_owned_process\.py --output ("+re.escape(args.run_root)+suffix+r") ")
            guards = {}
            for pid,(_,command) in records.items():
                match = marker.search(command)
                if match:
                    guards[pid] = match.group(1)
            roots = list(guards)
            if len(roots) > 1:
                raise ValueError("MULTIPLE_OWNED_GUARDS")
            owned = descendants(records, roots)
            rss = {}
            cpu_ticks = {}
            for pid in owned:
                try:
                    status = Path(f"/proc/{pid}/status").read_text()
                    match = re.search(r"^VmRSS:\s*(\d+)\s+kB", status, re.M)
                    if match:
                        rss[str(pid)] = int(match.group(1))*1024
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
                    cpu_ticks[str(pid)] = int(fields[11])+int(fields[12])
                except FileNotFoundError:
                    pass
            raw = read(["npu-smi", "info", "-t", "proc-mem", "-i", "0", "-c", "0"])
            hbm = proc_memory(raw)
            if not hbm and "Process" in raw:
                raise ValueError("HBM_OUTPUT_SCHEMA")
            item.update(status="SAMPLED" if roots else "NO_OWNED_GUARD", owned_host_pids=sorted(owned), owned_guard_paths=guards,
                        summed_owned_rss_bytes=sum(rss.values()), rss_by_host_pid=rss,
                        owned_hbm_mb_by_pid={str(pid):mb for pid,mb in hbm.items() if pid in owned},
                        other_process_hbm_mb=sum(mb for pid,mb in hbm.items() if pid not in owned),
                        owned_cpu_ticks_by_pid=cpu_ticks, clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
                        host_load_average=list(os.getloadavg()),
                        affinity_cpu_counters={line.split()[0]:list(map(int,line.split()[1:])) for line in Path("/proc/stat").read_text().splitlines()
                                               if line.split()[0] in {f"cpu{i}" for i in range(16)}},
                        npu0_usages_text=read(["npu-smi","info","-t","usages","-i","0","-c","0"]))
        except Exception as exc:
            item.update(status="MONITOR_ERROR", exception=repr(exc))
        print(json.dumps(item), flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
