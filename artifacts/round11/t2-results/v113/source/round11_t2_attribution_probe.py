"""Deep attribution of the frozen C128/MXFP4 layer-0 native worker path."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import struct
import time
import types


CAPTURE_SHA256 = "c837641468b9db93c422a3610af7a7ecc1f14d4bbb359841d69ac0dfd655dc6e"
BANK_SHA256 = "e9d6b27f352d501a9ed3737ec5d3ab140b9d1f5bda22d2e36ef9ef0bcc45b08b"
NATIVE_SHA256 = "614ae60f70ff70d1e47d7fd32e857f887f874b6248eb332172c8a637cb471a6d"
OUTPUT_SHA256 = "f5e189843493b9f42fcf724d9487ec5e1b9c630a6b0e780079cbc2bf6053f7c4"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRAINING_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
NUMA_NODES = [2, 3, 4, 7]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha256(tensor):
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("empty timing series")
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_us(values):
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "p05_us": percentile(values, 0.05),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
        "max_us": max(values),
        "mean_us": statistics.fmean(values),
    }


def summarize_perf_counts(rows):
    totals = {}
    coverage = {}
    for row in rows:
        running = row.get("time_running")
        enabled = row.get("time_enabled")
        if not isinstance(running, int) or running <= 0:
            raise ValueError("perf counter time_running must be positive")
        if not isinstance(enabled, int) or enabled < running:
            raise ValueError("perf counter time_enabled is invalid")
        event = row["event"]
        scaled = float(row["raw"]) * enabled / running
        totals[event] = totals.get(event, 0.0) + scaled
        coverage[event] = coverage.get(event, 0) + 1
    cycles = totals.get("cycles")
    instructions = totals.get("instructions")
    references = totals.get("cache_references")
    misses = totals.get("cache_misses")
    return {
        "scaled_totals": totals,
        "thread_coverage": coverage,
        "ipc": None if not cycles else instructions / cycles,
        "generic_cache_miss_ratio": None if not references or misses is None else misses / references,
    }


class PerfThreadCounters:
    """Count this process's existing threads without requiring the perf CLI."""

    # linux/perf_event.h generic events. Generic cache misses are not an LLC
    # attribution and are labelled accordingly in the final result.
    EVENTS = (
        ("cycles", 0, 0),
        ("instructions", 0, 1),
        ("cache_references", 0, 2),
        ("cache_misses", 0, 3),
        ("context_switches", 1, 3),
        ("cpu_migrations", 1, 4),
    )
    PERF_EVENT_IOC_ENABLE = 0x2400
    PERF_EVENT_IOC_DISABLE = 0x2401
    PERF_EVENT_IOC_RESET = 0x2403

    class Attr(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_uint32),
            ("size", ctypes.c_uint32),
            ("config", ctypes.c_uint64),
            ("sample_period", ctypes.c_uint64),
            ("sample_type", ctypes.c_uint64),
            ("read_format", ctypes.c_uint64),
            ("flags", ctypes.c_uint64),
            ("wakeup_events", ctypes.c_uint32),
            ("bp_type", ctypes.c_uint32),
            ("config1", ctypes.c_uint64),
            ("config2", ctypes.c_uint64),
            ("branch_sample_type", ctypes.c_uint64),
            ("sample_regs_user", ctypes.c_uint64),
            ("sample_stack_user", ctypes.c_uint32),
            ("clockid", ctypes.c_int32),
            ("sample_regs_intr", ctypes.c_uint64),
            ("aux_watermark", ctypes.c_uint32),
            ("sample_max_stack", ctypes.c_uint16),
            ("reserved_2", ctypes.c_uint16),
            ("aux_sample_size", ctypes.c_uint32),
            ("reserved_3", ctypes.c_uint32),
            ("sig_data", ctypes.c_uint64),
        ]

    def __init__(self, tids):
        machine = platform.machine().lower()
        syscall_number = 241 if machine in {"aarch64", "arm64"} else 298 if machine == "x86_64" else None
        self.rows = []
        self.open_errors = []
        self.fds = []
        self.tids = sorted(set(int(tid) for tid in tids))
        if syscall_number is None:
            self.open_errors.append({"error": f"unsupported perf_event_open architecture {machine}"})
            return
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        for tid in self.tids:
            for event, event_type, config in self.EVENTS:
                attr = self.Attr()
                attr.type = event_type
                attr.size = ctypes.sizeof(attr)
                attr.config = config
                attr.read_format = 3  # TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING
                attr.flags = (1 << 0) | (1 << 5) | (1 << 6)  # disabled, exclude kernel/hypervisor
                fd = int(libc.syscall(syscall_number, ctypes.byref(attr), tid, -1, -1, 0))
                if fd < 0:
                    number = ctypes.get_errno()
                    self.open_errors.append(
                        {"tid": tid, "event": event, "errno": number, "error": os.strerror(number)}
                    )
                else:
                    self.fds.append((fd, tid, event))

    def enable(self):
        for fd, _tid, _event in self.fds:
            fcntl.ioctl(fd, self.PERF_EVENT_IOC_RESET, 0)
            fcntl.ioctl(fd, self.PERF_EVENT_IOC_ENABLE, 0)

    def disable_read_close(self):
        if self.rows:
            return self.report()
        for fd, tid, event in self.fds:
            try:
                fcntl.ioctl(fd, self.PERF_EVENT_IOC_DISABLE, 0)
                raw, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
                self.rows.append(
                    {
                        "tid": tid,
                        "event": event,
                        "raw": raw,
                        "time_enabled": enabled,
                        "time_running": running,
                    }
                )
            except OSError as error:
                self.open_errors.append(
                    {"tid": tid, "event": event, "errno": error.errno, "error": str(error)}
                )
            finally:
                os.close(fd)
        self.fds = []
        return self.report()

    def close(self):
        for fd, _tid, _event in self.fds:
            os.close(fd)
        self.fds = []

    def report(self):
        complete = len(self.rows) == len(self.tids) * len(self.EVENTS) and not self.open_errors
        value = {
            "status": "PASS_ALL_EXISTING_THREADS" if complete else "PARTIAL_OR_UNAVAILABLE",
            "target_thread_count": len(self.tids),
            "opened_counter_count": len(self.rows),
            "open_errors": self.open_errors,
            "rows": self.rows,
            "llc_misses": "UNKNOWN_GENERIC_CACHE_EVENT_IS_NOT_LLC_SPECIFIC",
            "memory_controller_bandwidth": "UNKNOWN_NO_SUPPORTED_CONTROLLER_EVENT_MAPPING",
            "remote_numa_accesses": "UNKNOWN_NO_SUPPORTED_UNCORE_OR_NUMA_ACCESS_EVENT_MAPPING",
        }
        if self.rows:
            value["summary"] = summarize_perf_counts(self.rows)
        return value


def validate_admission(document, *, now=None):
    now = time.time() if now is None else now
    ended = document.get("ended_at")
    if (
        document.get("status") != "PASS"
        or document.get("windows") != 12
        or document.get("training_id") != TRAINING_ID
        or document.get("selected_physical_cores") != 160
        or not isinstance(ended, (int, float))
    ):
        raise ValueError("Round11 component admission identity mismatch")
    if ended > now + 5 or now - ended > 900:
        raise ValueError("Round11 component admission is stale")
    if float(document.get("min_mean_idle", 0)) <= 0.80:
        raise ValueError("Round11 component admission idle gate mismatch")
    return document


def thread_snapshot():
    root = Path(f"/proc/{os.getpid()}/task")
    result = {}
    for path in root.iterdir():
        try:
            tid = int(path.name)
            raw = (path / "stat").read_text()
            right = raw.rfind(")")
            rest = raw[right + 2 :].split()
            status = (path / "status").read_text().splitlines()
            allowed = next(
                line.split(":", 1)[1].strip()
                for line in status
                if line.startswith("Cpus_allowed_list:")
            )
            sched = [int(value) for value in (path / "schedstat").read_text().split()]
            result[str(tid)] = {
                "name": (path / "comm").read_text().strip(),
                "utime_ticks": int(rest[11]),
                "stime_ticks": int(rest[12]),
                "last_cpu": int(rest[36]),
                "allowed_cpus": allowed,
                "sched_runtime_ns": sched[0],
                "sched_wait_ns": sched[1],
                "sched_timeslices": sched[2],
            }
        except (FileNotFoundError, ProcessLookupError, StopIteration, ValueError):
            continue
    return result


def thread_delta(before, after):
    rows = []
    for tid in sorted(set(before) & set(after), key=int):
        old, new = before[tid], after[tid]
        row = {
            "tid": int(tid),
            "name": new["name"],
            "allowed_cpus": new["allowed_cpus"],
            "last_cpu": new["last_cpu"],
            "cpu_ticks": (new["utime_ticks"] + new["stime_ticks"]) - (old["utime_ticks"] + old["stime_ticks"]),
            "sched_runtime_ns": new["sched_runtime_ns"] - old["sched_runtime_ns"],
            "sched_wait_ns": new["sched_wait_ns"] - old["sched_wait_ns"],
            "sched_timeslices": new["sched_timeslices"] - old["sched_timeslices"],
        }
        if row["cpu_ticks"] or row["sched_runtime_ns"] or row["sched_wait_ns"]:
            rows.append(row)
    return rows


NUMA_COMPUTE_THREAD_RE = re.compile(r"^numa_[0-9]+_[tm]_")


def select_compute_thread_ids(snapshot):
    """Return only the four C128 subpools' worker/distributor threads.

    Each of the four TP partitions uses one named distributor thread as its
    local thread zero and 31 named InNumaPool workers.  torch_npu and Python
    runtime threads live in the same process, so process-wide counters would
    not describe the Kunpeng Expert kernel.
    """
    tids = sorted(
        int(tid)
        for tid, row in snapshot.items()
        if NUMA_COMPUTE_THREAD_RE.match(str(row.get("name", "")))
    )
    if len(tids) != 128:
        raise RuntimeError(f"expected exactly 128 named NUMA compute threads, found {len(tids)}")
    return tids


TP_RE = re.compile(
    r"\[KT_PHASE tp(?P<tp>\d+)\] n=(?P<n>\d+) avg/layer-call: "
    r"quant=(?P<quant>[0-9.]+)us gateup=(?P<gateup>[0-9.]+)us down=(?P<down>[0-9.]+)us"
)
MERGE_RE = re.compile(
    r"\[KT_PHASE merge\] n=(?P<n>\d+) avg/layer-call: merge=(?P<merge>[0-9.]+)us"
)
TASK_QUEUE_RE = re.compile(
    r"\[KT_TASK_QUEUE\] n=(?P<n>\d+) avg/task: "
    r"queue_admission_wait=(?P<wait>[0-9.]+)us task_execute=(?P<execute>[0-9.]+)us"
)


def parse_task_queue_log(raw, *, expected_calls):
    latest = None
    for line in raw.splitlines():
        if match := TASK_QUEUE_RE.search(line):
            latest = {
                "calls": int(match.group("n")),
                "queue_admission_wait_us_per_group": float(match.group("wait")),
                "task_execute_us_per_group": float(match.group("execute")),
            }
    if latest is None:
        raise RuntimeError("native TaskQueue timing evidence missing")
    if latest["calls"] != expected_calls:
        raise RuntimeError("native TaskQueue timing call count mismatch")
    return latest


def parse_phase_log(raw, *, expected_tp_calls, expected_merge_calls, qlen):
    latest = {}
    merge = None
    for line in raw.splitlines():
        if match := TP_RE.search(line):
            latest[int(match.group("tp"))] = {
                "calls": int(match.group("n")),
                "quant_us_per_row": float(match.group("quant")),
                "gateup_activation_us_per_row": float(match.group("gateup")),
                "down_weighted_sum_us_per_row": float(match.group("down")),
            }
        if match := MERGE_RE.search(line):
            merge = {
                "calls": int(match.group("n")),
                "merge_publication_us_per_group": float(match.group("merge")),
            }
    if set(latest) != {0, 1, 2, 3}:
        raise RuntimeError(f"native phase evidence missing TP partitions: {sorted(latest)}")
    if any(row["calls"] != expected_tp_calls for row in latest.values()):
        raise RuntimeError("native TP phase call count mismatch")
    if merge is None or merge["calls"] != expected_merge_calls:
        raise RuntimeError("native merge phase call count mismatch")
    totals = {
        tp: qlen
        * (
            row["quant_us_per_row"]
            + row["gateup_activation_us_per_row"]
            + row["down_weighted_sum_us_per_row"]
        )
        for tp, row in latest.items()
    }
    return {
        "tp_partitions": {str(key): value for key, value in latest.items()},
        "critical_tp_compute_us_per_group": max(totals.values()),
        "fastest_tp_compute_us_per_group": min(totals.values()),
        "tp_partition_imbalance_ratio": max(totals.values()) / max(min(totals.values()), 1e-30),
        "merge": merge,
        "native_compute_plus_merge_us_per_group": max(totals.values())
        + merge["merge_publication_us_per_group"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-log", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--attribution-build-receipt", type=Path)
    parser.add_argument("--hardware-counters", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    if args.output.exists() or args.phase_log.exists():
        raise FileExistsError("attribution evidence must be fresh")
    if not 100 <= args.iterations <= 5000:
        raise ValueError("bounded attribution iterations must be 100..5000")
    warmups = 5
    phase_interval = warmups + args.iterations
    os.environ["KT_MOE_PHASE_TIMING"] = "1"
    os.environ["KT_MOE_PHASE_TIMING_INTERVAL"] = str(phase_interval)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "scope": "ROUND11_T2_REAL_LAYER0_C128_MXFP4_DEEP_ATTRIBUTION",
        "iterations": args.iterations,
        "warmups": warmups,
        "tokens": 3,
        "numa_nodes": NUMA_NODES,
        "cpu_workers": 128,
        "threadpool_count": 4,
        "profiler": "NATIVE_RUNTIME_PHASE_TIMING_AND_PROC_THREAD_ACCOUNTING",
    }
    try:
        admission = validate_admission(json.loads(args.admission.read_text()))
        report["admission"] = {
            "sha256": digest(args.admission),
            "ended_at": admission["ended_at"],
            "windows": admission["windows"],
            "min_mean_idle": admission["min_mean_idle"],
            "training_id": admission["training_id"],
        }
        import torch
        import kt_kernel
        from kt_kernel import KTMoEWrapper

        torch.set_num_threads(2)
        native_path = Path(kt_kernel.kt_kernel_ext.__file__).resolve(strict=True)
        build_receipt = None
        expected_native_sha256 = NATIVE_SHA256
        binary_role = "FROZEN_HFINAL_NATIVE"
        if args.attribution_build_receipt is not None:
            build_receipt = json.loads(args.attribution_build_receipt.read_text())
            if (
                build_receipt.get("status") != "PASS_ROUND11_T2_ATTRIBUTION_BUILD"
                or build_receipt.get("baseline_native_sha256") != NATIVE_SHA256
                or not isinstance(build_receipt.get("attribution_native_sha256"), str)
            ):
                raise ValueError("attribution-only native build receipt mismatch")
            expected_native_sha256 = build_receipt["attribution_native_sha256"]
            binary_role = "ATTRIBUTION_ONLY_TASK_QUEUE_TIMESTAMPS"
        identities = {
            "capture_sha256": digest(args.capture),
            "bank_sha256": digest(args.bank),
            "native_sha256": digest(native_path),
        }
        if identities != {
            "capture_sha256": CAPTURE_SHA256,
            "bank_sha256": BANK_SHA256,
            "native_sha256": expected_native_sha256,
        }:
            raise ValueError("Round11 T2 frozen input/native identity changed")
        report.update(
            identities,
            native_path=str(native_path),
            binary_role=binary_role,
            attribution_build_receipt=(
                str(args.attribution_build_receipt) if args.attribution_build_receipt is not None else None
            ),
        )
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        group = payload["groups"][0]
        hidden = group["hidden_states"].contiguous()
        ids = group["topk_logical_ids"].to(torch.int64).contiguous()
        weights = group["topk_weights"].float().contiguous()
        if hidden.shape != (3, 2560) or ids.shape != (3, 10) or weights.shape != (3, 10):
            raise ValueError("real attribution fixture dimensions changed")

        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=512,
            num_experts_per_tok=10,
            hidden_size=2560,
            moe_intermediate_size=640,
            gpu_experts_mask=torch.zeros(512, dtype=torch.bool),
            cpuinfer_threads=128,
            threadpool_count=4,
            weight_path=str(args.bank),
            chunked_prefill_size=32,
            max_deferred_experts_per_token=0,
            method="LLAMAFILE",
            numa_nodes=NUMA_NODES,
        )
        wrapper.load_weights()
        if build_receipt is not None:
            os.environ["KT_TASK_QUEUE_TIMING"] = "1"
            os.environ["KT_TASK_QUEUE_TIMING_INTERVAL"] = str(phase_interval)
        initial_threads = thread_snapshot()
        compute_thread_ids = select_compute_thread_ids(initial_threads)
        perf_counters = (
            PerfThreadCounters(compute_thread_ids)
            if args.hardware_counters == "ON"
            else None
        )

        timings = {"prepare": [], "submit_parent": [], "sync_parent": [], "publication": [], "total": []}
        original_prepare = wrapper._prepare_forward_cpu_buffers
        original_publication = wrapper.copy_forward_output_to_device

        def timed_prepare(self, *values, **kwargs):
            started = time.perf_counter_ns()
            result = original_prepare(*values, **kwargs)
            timings["prepare"].append((time.perf_counter_ns() - started) / 1e3)
            return result

        def timed_publication(self, *values, **kwargs):
            started = time.perf_counter_ns()
            result = original_publication(*values, **kwargs)
            timings["publication"].append((time.perf_counter_ns() - started) / 1e3)
            return result

        wrapper._prepare_forward_cpu_buffers = types.MethodType(timed_prepare, wrapper)
        wrapper.copy_forward_output_to_device = types.MethodType(timed_publication, wrapper)

        saved_stderr = os.dup(2)
        phase_stream = args.phase_log.open("xb", buffering=0)
        os.dup2(phase_stream.fileno(), 2)
        hashes = []
        before_threads = None
        bulk_started = None
        try:
            total_calls = warmups + args.iterations
            for index in range(total_calls):
                if index == warmups:
                    for values in timings.values():
                        values.clear()
                    before_threads = thread_snapshot()
                    if perf_counters is not None:
                        perf_counters.enable()
                    bulk_started = time.perf_counter_ns()
                total_started = time.perf_counter_ns()
                submit_started = time.perf_counter_ns()
                wrapper.submit_forward(hidden, ids, weights, None)
                timings["submit_parent"].append((time.perf_counter_ns() - submit_started) / 1e3)
                sync_started = time.perf_counter_ns()
                output = wrapper.sync_forward(hidden, None)
                timings["sync_parent"].append((time.perf_counter_ns() - sync_started) / 1e3)
                timings["total"].append((time.perf_counter_ns() - total_started) / 1e3)
                if index >= warmups and (index - warmups < 5 or index == total_calls - 1):
                    hashes.append(tensor_sha256(output))
            bulk_ended = time.perf_counter_ns()
            hardware_counters = (
                perf_counters.disable_read_close()
                if perf_counters is not None
                else {"status": "DISABLED_FOR_TIMING_ISOLATION"}
            )
            after_threads = thread_snapshot()
            ctypes.CDLL(None).fflush(None)
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            phase_stream.close()

        if set(hashes) != {OUTPUT_SHA256}:
            raise AssertionError("attribution output differs from frozen same-precision result")
        if any(len(values) != args.iterations for values in timings.values()):
            raise RuntimeError("exclusive timing coverage mismatch")
        prepare = timings["prepare"]
        publication = timings["publication"]
        submit_exclusive = [parent - child for parent, child in zip(timings["submit_parent"], prepare)]
        exposed_wait = [parent - child for parent, child in zip(timings["sync_parent"], publication)]
        if min(submit_exclusive + exposed_wait) < 0:
            raise RuntimeError("child timing exceeded parent span")
        phase = parse_phase_log(
            args.phase_log.read_text(errors="replace"),
            expected_tp_calls=3 * phase_interval,
            expected_merge_calls=phase_interval,
            qlen=3,
        )
        queue_separation = (
            parse_task_queue_log(args.phase_log.read_text(errors="replace"), expected_calls=phase_interval)
            if build_receipt is not None
            else "BOUNDED_RESIDUAL_ONLY_WITH_FROZEN_BINARY"
        )
        report.update(
            status="PASS_ROUND11_T2_DEEP_ATTRIBUTION",
            output_sha256=hashes[-1],
            repeat_hash_exact=True,
            parent_and_exclusive_us={
                "total_parent": summarize_us(timings["total"]),
                "submit_parent": summarize_us(timings["submit_parent"]),
                "pinned_gather_and_metadata": summarize_us(prepare),
                "enqueue_and_task_construction_exclusive": summarize_us(submit_exclusive),
                "sync_parent": summarize_us(timings["sync_parent"]),
                "exposed_queue_and_compute_wait_exclusive": summarize_us(exposed_wait),
                "result_publication": summarize_us(publication),
            },
            native_phases=phase,
            queue_wait_separation=queue_separation,
            thread_accounting={
                "elapsed_ns": bulk_ended - bulk_started,
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "compute_thread_ids": compute_thread_ids,
                "rows": thread_delta(before_threads, after_threads),
            },
            hardware_counters=hardware_counters,
        )
    except BaseException as error:
        report.update(status="FAIL_ROUND11_T2_ATTRIBUTION", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if "perf_counters" in locals() and perf_counters is not None:
            perf_counters.close()
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["ended_at"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["iterations"], flush=True)


if __name__ == "__main__":
    main()
