"""Create an attribution-only kt-kernel source tree with exact TaskQueue events.

The patch is deliberately limited to TaskQueue.  It does not change queue
ordering, worker-pool code, MoE math, stream synchronization, or the Python
ABI.  When ``KT_ROOT_TASK_TRACE_PATH`` is unset the only added work is a
single null environment lookup per enqueue/sync call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


TASK_QUEUE_H_SHA256 = "2c8d92bbd3a278e66918f0c6fec5413e4f004740f17051fb739524b0b3811248"
TASK_QUEUE_CPP_SHA256 = "71b30080568aa3a3beffff3f35d11fdc15f8bfaeadeffe4430ec90aa20b5d5ae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one source anchor, found {count}")
    return source.replace(old, new)


def patch_header(source: str) -> str:
    source = replace_once(
        source,
        "#include <condition_variable>\n#include <exception>",
        "#include <condition_variable>\n#include <cstdint>\n#include <exception>",
        "header include",
    )
    return replace_once(
        source,
        "    std::function<void()> task;\n"
        "    std::atomic<Node*> next;\n"
        "    Node() : task(nullptr), next(nullptr) {}\n"
        "    Node(const std::function<void()>& t) : task(t), next(nullptr) {}",
        "    std::function<void()> task;\n"
        "    std::atomic<Node*> next;\n"
        "    uint64_t trace_sequence;\n"
        "    uint64_t admitted_ns;\n"
        "    Node() : task(nullptr), next(nullptr), trace_sequence(0), admitted_ns(0) {}\n"
        "    Node(const std::function<void()>& t, uint64_t sequence, uint64_t admitted)\n"
        "        : task(t), next(nullptr), trace_sequence(sequence), admitted_ns(admitted) {}",
        "node trace fields",
    )


def patch_source(source: str) -> str:
    source = replace_once(
        source,
        "#include <chrono>\n#include <iostream>\n#include <thread>",
        "#include <chrono>\n#include <cstdio>\n#include <cstdlib>\n#include <iostream>\n#include <mutex>\n#include <thread>\n\n"
        "namespace {\n"
        "uint64_t kt_root_trace_now_ns() {\n"
        "  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
        "      std::chrono::steady_clock::now().time_since_epoch()).count();\n"
        "}\n"
        "struct KtRootTaskTrace {\n"
        "  std::mutex mutex;\n"
        "  std::atomic<uint64_t> task_sequence{0};\n"
        "  std::atomic<uint64_t> sync_sequence{0};\n"
        "  FILE* stream = nullptr;\n"
        "  bool initialized = false;\n"
        "  FILE* get() {\n"
        "    std::lock_guard<std::mutex> lock(mutex);\n"
        "    if (!initialized) {\n"
        "      initialized = true;\n"
        "      const char* path = std::getenv(\"KT_ROOT_TASK_TRACE_PATH\");\n"
        "      if (path && path[0]) {\n"
        "        stream = std::fopen(path, \"a\");\n"
        "        if (stream) std::setvbuf(stream, nullptr, _IOFBF, 1024 * 1024);\n"
        "      }\n"
        "    }\n"
        "    return stream;\n"
        "  }\n"
        "  void event(const char* kind, uint64_t sequence, uint64_t now_ns,\n"
        "             uint64_t admitted_ns, uint64_t start_ns, size_t pending_count,\n"
        "             size_t allow_pending) {\n"
        "    FILE* output = get();\n"
        "    if (!output) return;\n"
        "    std::lock_guard<std::mutex> lock(mutex);\n"
        "    std::fprintf(output,\n"
        "      \"{\\\"schema\\\":1,\\\"event\\\":\\\"%s\\\",\\\"sequence\\\":%llu,\"\n"
        "      \"\\\"now_ns\\\":%llu,\\\"admitted_ns\\\":%llu,\\\"start_ns\\\":%llu,\"\n"
        "      \"\\\"pending\\\":%zu,\\\"allow_pending\\\":%zu}\\n\",\n"
        "      kind, (unsigned long long)sequence, (unsigned long long)now_ns,\n"
        "      (unsigned long long)admitted_ns, (unsigned long long)start_ns,\n"
        "      pending_count, allow_pending);\n"
        "  }\n"
        "  ~KtRootTaskTrace() { if (stream) std::fclose(stream); }\n"
        "};\n"
        "KtRootTaskTrace g_kt_root_task_trace;\n"
        "}\n",
        "trace implementation",
    )
    source = replace_once(
        source,
        "  pending.fetch_add(1, std::memory_order_acq_rel);\n"
        "  Node* node = new Node(task);",
        "  pending.fetch_add(1, std::memory_order_acq_rel);\n"
        "  FILE* trace_stream = g_kt_root_task_trace.get();\n"
        "  const uint64_t sequence = trace_stream\n"
        "      ? g_kt_root_task_trace.task_sequence.fetch_add(1, std::memory_order_relaxed) + 1 : 0;\n"
        "  const uint64_t admitted_ns = trace_stream ? kt_root_trace_now_ns() : 0;\n"
        "  Node* node = new Node(task, sequence, admitted_ns);\n"
        "  if (trace_stream) g_kt_root_task_trace.event(\n"
        "      \"enqueue\", sequence, admitted_ns, admitted_ns, 0,\n"
        "      pending.load(std::memory_order_acquire), 0);",
        "enqueue trace",
    )
    source = replace_once(
        source,
        "void TaskQueue::sync(size_t allow_n_pending) {\n"
        "  std::exception_ptr task_exception;\n"
        "  {",
        "void TaskQueue::sync(size_t allow_n_pending) {\n"
        "  std::exception_ptr task_exception;\n"
        "  FILE* trace_stream = g_kt_root_task_trace.get();\n"
        "  const uint64_t sync_sequence = trace_stream\n"
        "      ? g_kt_root_task_trace.sync_sequence.fetch_add(1, std::memory_order_relaxed) + 1 : 0;\n"
        "  const uint64_t sync_start_ns = trace_stream ? kt_root_trace_now_ns() : 0;\n"
        "  if (trace_stream) g_kt_root_task_trace.event(\n"
        "      \"sync_begin\", sync_sequence, sync_start_ns, 0, 0,\n"
        "      pending.load(std::memory_order_acquire), allow_n_pending);\n"
        "  {",
        "sync begin trace",
    )
    source = replace_once(
        source,
        "  if (task_exception) std::rethrow_exception(task_exception);\n}",
        "  if (trace_stream) g_kt_root_task_trace.event(\n"
        "      \"sync_end\", sync_sequence, kt_root_trace_now_ns(), 0, sync_start_ns,\n"
        "      pending.load(std::memory_order_acquire), allow_n_pending);\n"
        "  if (task_exception) std::rethrow_exception(task_exception);\n}",
        "sync end trace",
    )
    return replace_once(
        source,
        "      if (next->task) {\n"
        "        try {\n"
        "          next->task();\n"
        "        } catch (...) {\n"
        "          task_exception = std::current_exception();\n"
        "        }\n"
        "      }",
        "      if (next->task) {\n"
        "        const uint64_t task_start_ns = next->admitted_ns ? kt_root_trace_now_ns() : 0;\n"
        "        try {\n"
        "          next->task();\n"
        "        } catch (...) {\n"
        "          task_exception = std::current_exception();\n"
        "        }\n"
        "        if (next->admitted_ns) g_kt_root_task_trace.event(\n"
        "            \"task_complete\", next->trace_sequence, kt_root_trace_now_ns(),\n"
        "            next->admitted_ns, task_start_ns,\n"
        "            pending.load(std::memory_order_acquire), 0);\n"
        "      }",
        "task completion trace",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("trace source and receipt must be fresh")
    identities = {
        "task_queue_h": sha256(args.source / "kt-kernel/cpu_backend/task_queue.h"),
        "task_queue_cpp": sha256(args.source / "kt-kernel/cpu_backend/task_queue.cpp"),
    }
    expected = {
        "task_queue_h": TASK_QUEUE_H_SHA256,
        "task_queue_cpp": TASK_QUEUE_CPP_SHA256,
    }
    if identities != expected:
        raise ValueError(f"frozen source identity changed: {identities}")
    shutil.copytree(args.source, args.output, symlinks=True)
    header = args.output / "kt-kernel/cpu_backend/task_queue.h"
    implementation = args.output / "kt-kernel/cpu_backend/task_queue.cpp"
    header.write_text(patch_header(header.read_text()))
    implementation.write_text(patch_source(implementation.read_text()))
    receipt = {
        "schema": 1,
        "status": "BUILT_KT_ROOT_CAUSE_TASK_TRACE_SOURCE",
        "created_at": time.time(),
        "baseline_source": str(args.source),
        "trace_source": str(args.output),
        "baseline_sha256": identities,
        "trace_sha256": {
            "task_queue_h": sha256(header),
            "task_queue_cpp": sha256(implementation),
        },
        "semantic_scope": "ENV_GATED_TASK_QUEUE_TIMESTAMPS_ONLY",
        "production_synchronization_changes": 0,
        "optimization_candidate": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
