"""Clone the frozen native source and add env-gated TaskQueue timing only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


HEADER_SHA256 = "2c8d92bbd3a278e66918f0c6fec5413e4f004740f17051fb739524b0b3811248"
SOURCE_SHA256 = "71b30080568aa3a3beffff3f35d11fdc15f8bfaeadeffe4430ec90aa20b5d5ae"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError(f"{label} precondition mismatch")
    return text.replace(old, new)


def patch_header(text):
    text = replace_once(
        text,
        "#include <condition_variable>\n#include <exception>",
        "#include <condition_variable>\n#include <cstdint>\n#include <exception>",
        "TaskQueue header include",
    )
    return replace_once(
        text,
        "    std::function<void()> task;\n"
        "    std::atomic<Node*> next;\n"
        "    Node() : task(nullptr), next(nullptr) {}\n"
        "    Node(const std::function<void()>& t) : task(t), next(nullptr) {}",
        "    std::function<void()> task;\n"
        "    std::atomic<Node*> next;\n"
        "    uint64_t admitted_ns;\n"
        "    Node() : task(nullptr), next(nullptr), admitted_ns(0) {}\n"
        "    Node(const std::function<void()>& t, uint64_t admitted)\n"
        "        : task(t), next(nullptr), admitted_ns(admitted) {}",
        "TaskQueue Node",
    )


def patch_source(text):
    text = replace_once(
        text,
        "#include <chrono>\n#include <iostream>\n#include <thread>",
        "#include <chrono>\n#include <cstdio>\n#include <cstdlib>\n#include <iostream>\n#include <thread>\n\n"
        "namespace {\n"
        "uint64_t kt_task_queue_now_ns() {\n"
        "  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
        "      std::chrono::steady_clock::now().time_since_epoch()).count();\n"
        "}\n"
        "bool kt_task_queue_timing_on() {\n"
        "  return std::getenv(\"KT_TASK_QUEUE_TIMING\") != nullptr;\n"
        "}\n"
        "uint64_t kt_task_queue_interval() {\n"
        "  const char* raw = std::getenv(\"KT_TASK_QUEUE_TIMING_INTERVAL\");\n"
        "  const uint64_t value = raw ? std::strtoull(raw, nullptr, 10) : 0;\n"
        "  return value ? value : 4096;\n"
        "}\n"
        "struct KtTaskQueueTiming {\n"
        "  uint64_t count = 0;\n"
        "  uint64_t queue_admission_wait_ns = 0;\n"
        "  uint64_t task_execute_ns = 0;\n"
        "};\n"
        "KtTaskQueueTiming g_kt_task_queue_timing;\n"
        "}\n",
        "TaskQueue timing helpers",
    )
    text = replace_once(
        text,
        "  pending.fetch_add(1, std::memory_order_acq_rel);\n"
        "  Node* node = new Node(task);",
        "  pending.fetch_add(1, std::memory_order_acq_rel);\n"
        "  const uint64_t admitted_ns = kt_task_queue_timing_on() ? kt_task_queue_now_ns() : 0;\n"
        "  Node* node = new Node(task, admitted_ns);",
        "TaskQueue admission timestamp",
    )
    return replace_once(
        text,
        "      if (next->task) {\n"
        "        try {\n"
        "          next->task();\n"
        "        } catch (...) {\n"
        "          task_exception = std::current_exception();\n"
        "        }\n"
        "      }",
        "      if (next->task) {\n"
        "        const uint64_t task_start_ns = next->admitted_ns ? kt_task_queue_now_ns() : 0;\n"
        "        try {\n"
        "          next->task();\n"
        "        } catch (...) {\n"
        "          task_exception = std::current_exception();\n"
        "        }\n"
        "        if (next->admitted_ns) {\n"
        "          const uint64_t task_end_ns = kt_task_queue_now_ns();\n"
        "          auto& timing = g_kt_task_queue_timing;\n"
        "          timing.count++;\n"
        "          timing.queue_admission_wait_ns += task_start_ns - next->admitted_ns;\n"
        "          timing.task_execute_ns += task_end_ns - task_start_ns;\n"
        "          if (timing.count % kt_task_queue_interval() == 0) {\n"
        "            std::fprintf(stderr,\n"
        "                \"[KT_TASK_QUEUE] n=%llu avg/task: queue_admission_wait=%.1fus task_execute=%.1fus\\n\",\n"
        "                (unsigned long long)timing.count,\n"
        "                timing.queue_admission_wait_ns / 1e3 / timing.count,\n"
        "                timing.task_execute_ns / 1e3 / timing.count);\n"
        "          }\n"
        "        }\n"
        "      }",
        "TaskQueue worker timing",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("attribution source/receipt must be fresh")
    header = args.source / "kt-kernel/cpu_backend/task_queue.h"
    source = args.source / "kt-kernel/cpu_backend/task_queue.cpp"
    identities = {"task_queue_h": sha256(header), "task_queue_cpp": sha256(source)}
    if identities != {"task_queue_h": HEADER_SHA256, "task_queue_cpp": SOURCE_SHA256}:
        raise ValueError("frozen native TaskQueue source identity changed")
    shutil.copytree(args.source, args.output, symlinks=True)
    out_header = args.output / "kt-kernel/cpu_backend/task_queue.h"
    out_source = args.output / "kt-kernel/cpu_backend/task_queue.cpp"
    out_header.write_text(patch_header(out_header.read_text()))
    out_source.write_text(patch_source(out_source.read_text()))
    receipt = {
        "schema": 1,
        "status": "BUILT_ROUND11_T2_ATTRIBUTION_ONLY_SOURCE",
        "created_at": time.time(),
        "source_root": str(args.source),
        "output_root": str(args.output),
        "baseline_files_sha256": identities,
        "attribution_files_sha256": {
            "task_queue_h": sha256(out_header),
            "task_queue_cpp": sha256(out_source),
        },
        "semantic_scope": "ENV_GATED_QUEUE_ADMISSION_WAIT_AND_TASK_EXECUTE_TIMESTAMPS_ONLY",
        "optimization_candidate": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    main()
