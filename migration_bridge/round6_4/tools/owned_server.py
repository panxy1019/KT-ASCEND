"""Small lifecycle adapter around the existing project-only process-group guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_compact_controls import OLD, ensure_free_port


class OwnedServer:
    def __init__(self, root, profile, batch=1, capture=False, timeout=7200, extra_env=None, capacity=True):
        self.root, self.profile, self.batch, self.capture = Path(root), profile, batch, capture
        if profile not in ("D1", "D1_NODEFER") or batch not in (1, 2, 4):
            raise ValueError("UNSUPPORTED_PROFILE")
        if not str(self.root).startswith("/workspace/results/r64-"):
            raise ValueError("DEDICATED_ROOT_REQUIRED")
        self.timeout, self.extra_env = timeout, extra_env or {}
        self.capacity = capacity
        if not capacity and batch != 1:
            raise ValueError("ORIGINAL_CAPACITY_IS_BATCH1_ONLY")
        self.server = self.log = None
        self.report = dict(profile=profile, batch_cap=batch, capture=capture, capacity_adapter=capacity, started_unix=time.time(), status="STARTING")

    def __enter__(self):
        ensure_free_port()
        self.root.mkdir(parents=True, exist_ok=False)
        env = {k:v for k,v in os.environ.items() if not k.startswith(("KT_R6_", "KT_R61_", "KT_R62_", "KT_R63_", "KT_DEBUG_", "KT_R64_"))}
        adapter = Path(__file__).with_name("capacity_python_adapter.py")
        env.update(ASCEND_RT_VISIBLE_DEVICES="0", SGLANG_APPLY_CONFIG_BACKUP="none",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="false",
                   KT_R6_ENABLE_ROUTE_RECORDER="1", KT_R6_ENABLE_TENSOR_DUMP=str(int(self.capture)),
                   KT_R6_PYTHON=str(adapter) if self.capacity else sys.executable, KT_R64_BATCH_CAP=str(self.batch),
                   KT_R64_RESOLVED_COMMAND=str(self.root/"resolved-command.json"))
        if self.capture:
            env.update(KT_R62_COMPLETE_FORWARD_DUMP="1", KT_R63_COMPACT_CAPTURE="1",
                       KT_R63_FULL_DIAGNOSTIC_SEQUENCES="0", KT_R63_CAPTURE_INDEX=str(self.root/"index.jsonl"),
                       KT_R62_RUN_METADATA=str(self.root/"provenance.json"))
        else:
            (self.root/"tensors").mkdir()
        env.update(self.extra_env)
        self.env = env
        self.log = (self.root/"launcher.log").open("x")
        try:
            subprocess.run([sys.executable, str(OLD/"prepare_run_metadata.py"), "--mode", "defer1" if self.profile == "D1" else "defer0",
                "--output", str(self.root/"provenance.json"), "--artifact", str(adapter), "--artifact", str(Path(__file__)),
                "--artifact", "/workspace/results/placements/placement_d1.pt",
                "--artifact", "/workspace/results/gguf/deepseek-v2-lite-round6-d3-f32.gguf",
                "--artifact", "/workspace/models/DeepSeek-V2-Lite-604d5664/tokenizer.json",
                "--artifact", "/workspace/kt-src/third_party/sglang/python/sglang/srt/model_executor/model_runner.py"],
                env=env, check=True, timeout=600, stdout=self.log, stderr=subprocess.STDOUT)
            meta = json.loads((self.root/"provenance.json").read_text())
            reference = json.loads(Path("/workspace/results/r64-preflight-v1/provenance.json").read_text())
            reference["file_sha256"]["/workspace/kt-src/third_party/sglang/python/sglang/srt/model_executor/model_runner.py"] = "71b1daaf2a78afde91794963eb5c6a0779bb6ed931841e0a56f8b8bf34ce88d9"
            for name, sha in meta["file_sha256"].items():
                if name in reference["file_sha256"] and sha != reference["file_sha256"][name]:
                    raise ValueError(f"ACTUAL_SOURCE_DRIFT:{name}")
            command = [sys.executable, str(OLD/"run_owned_process.py"), "--output", str(self.root/"guard"),
                       "--timeout-seconds", str(self.timeout), "--", "taskset", "-c", "0-15", "bash",
                       "/workspace/kt-src/migration_bridge/round6/tools/launch_round6_profile.sh", self.profile, str(self.root), "31000"]
            self.server = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            self.report.update(owned_guard_pid=self.server.pid, command=command)
            subprocess.run([sys.executable, str(OLD/"wait_capture_ready.py"), "--log", str(self.root/"guard/process.log"),
                            "--captures", str(self.root/"tensors"), "--output", str(self.root/"readiness.json")],
                           env=env, check=True, timeout=920)
            self.report["status"] = "READY"
            return self
        except BaseException:
            self.close(failed=True)
            raise

    def close(self, failed=False):
        try:
            if self.server is not None and self.server.poll() is None:
                self.server.send_signal(signal.SIGTERM)
                self.server.wait(timeout=60)
            guard = self.root/"guard/result.json"
            if guard.exists():
                self.report["guard"] = json.loads(guard.read_text())
            expected = self.report.get("guard", {}).get("reason") == "CANCELLED"
            self.report.update(status="FAILED" if failed or not expected else "PLANNED_STOP", ended_unix=time.time())
            if not failed and not expected:
                raise RuntimeError("UNEXPECTED_OWNED_SERVER_TERMINATION")
        except BaseException as exc:
            self.report.update(status="FAILED", cleanup_exception=repr(exc))
            raise
        finally:
            (self.root/"lifecycle.json").write_text(json.dumps(self.report, indent=2)+"\n")
            if self.log:
                self.log.close()

    def __exit__(self, kind, value, traceback):
        self.close(failed=kind is not None)
        return False
