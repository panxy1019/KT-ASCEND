"""Small lifecycle adapter around the existing project-only process-group guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_compact_controls import OLD, ensure_free_port

STREAM_OVERRIDES = ("SGLANG_KT_HYBRID_NO_CPU_STREAM", "SGLANG_KT_HYBRID_TIMING",
                    "SGLANG_KT_HYBRID_TIMING_DEEP", "SGLANG_KT_BYPASS_GPU_MOE",
                    "KT_FORCE_SYNC_SUBMIT", "KT_EXTERNAL_NPU_REPORT_SUBSCRIBER")


def validate_stream_environment(env):
    values = {key:env.get(key) for key in STREAM_OVERRIDES}
    if any(value not in (None,"","0") for value in values.values()):
        raise ValueError("UNFROZEN_STREAM_OR_TIMING_OVERRIDE")
    return values


class OwnedServer:
    def __init__(self, root, profile, batch=1, capture=False, timeout=7200, extra_env=None, capacity=True, performance=False, tokenizer_compat=False):
        self.root, self.profile, self.batch, self.capture = Path(root), profile, batch, capture
        if profile not in ("D1", "D1_NODEFER") or batch not in (1, 2, 4):
            raise ValueError("UNSUPPORTED_PROFILE")
        if not str(self.root).startswith("/workspace/results/r64-"):
            raise ValueError("DEDICATED_ROOT_REQUIRED")
        self.timeout, self.extra_env = timeout, extra_env or {}
        self.capacity = capacity
        self.performance = performance
        self.tokenizer_compat = tokenizer_compat
        if tokenizer_compat and not capacity:
            raise ValueError("TOKENIZER_COMPAT_REQUIRES_EXPLICIT_ADAPTER")
        if performance and (capture or not capacity):
            raise ValueError("PERFORMANCE_REQUIRES_CAPACITY_AND_OBSERVER_OFF")
        if performance and self.extra_env:
            raise ValueError("PERFORMANCE_EXTRA_ENV_NOT_ALLOWED")
        if not capacity and batch != 1:
            raise ValueError("ORIGINAL_CAPACITY_IS_BATCH1_ONLY")
        self.server = self.log = None
        self.report = dict(profile=profile, batch_cap=batch, capture=capture, capacity_adapter=capacity,
                           performance_observer_off=performance, tokenizer_compatibility_overlay=tokenizer_compat,
                           started_unix=time.time(), status="STARTING")

    def __enter__(self):
        ensure_free_port()
        self.root.mkdir(parents=True, exist_ok=False)
        env = {k:v for k,v in os.environ.items() if not k.startswith(("KT_R6_", "KT_R61_", "KT_R62_", "KT_R63_", "KT_DEBUG_", "KT_R64_"))}
        adapter = Path(__file__).with_name("capacity_python_adapter.py")
        env.update(ASCEND_RT_VISIBLE_DEVICES="0", SGLANG_APPLY_CONFIG_BACKUP="none",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="false",
                   KT_R6_ENABLE_ROUTE_RECORDER="0" if self.performance else "1", KT_R6_ENABLE_TENSOR_DUMP=str(int(self.capture)),
                   KT_R64_PERFORMANCE=str(int(self.performance)),
                   KT_R64_TOKENIZER_COMPAT=str(int(self.tokenizer_compat)),
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
        self.report["stream_overrides"] = validate_stream_environment(env)
        self.log = (self.root/"launcher.log").open("x")
        try:
            subprocess.run([sys.executable, str(OLD/"prepare_run_metadata.py"), "--mode", "defer1" if self.profile == "D1" else "defer0",
                "--output", str(self.root/"provenance.json"), "--artifact", str(adapter), "--artifact", str(Path(__file__)),
                "--artifact", str(Path(__file__).with_name("run_owned_process.py")),
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
            command = [sys.executable, str(Path(__file__).with_name("run_owned_process.py")), "--output", str(self.root/"guard"),
                       "--timeout-seconds", str(self.timeout), "--", "taskset", "-c", "0-15", "bash",
                       "/workspace/kt-src/migration_bridge/round6/tools/launch_round6_profile.sh", self.profile, str(self.root), "31000"]
            self.server = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            self.report.update(owned_guard_pid=self.server.pid, command=command)
            ready = subprocess.Popen([sys.executable, str(OLD/"wait_capture_ready.py"), "--log", str(self.root/"guard/process.log"),
                            "--captures", str(self.root/"tensors"), "--output", str(self.root/"readiness.json")],
                           env=env)
            try:
                deadline = time.monotonic()+920
                while ready.poll() is None:
                    if self.server.poll() is not None:
                        raise RuntimeError("OWNED_SERVER_EXITED_BEFORE_READY")
                    if time.monotonic() > deadline:
                        raise TimeoutError("OWNED_SERVER_READINESS_TIMEOUT")
                    time.sleep(1)
                if ready.returncode:
                    raise subprocess.CalledProcessError(ready.returncode, ready.args)
            finally:
                if ready.poll() is None:
                    ready.terminate()
                    ready.wait(timeout=10)
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
            reaping = self.root/"guard/reaping.json"
            if reaping.exists():
                self.report["reaping"] = json.loads(reaping.read_text())
                if self.report["reaping"]["status"] != "OWNED_DESCENDANTS_REAPED":
                    raise RuntimeError("OWNED_DESCENDANTS_NOT_REAPED")
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
