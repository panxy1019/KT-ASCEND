import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    return json.loads(
        (ROOT.parent / "round14a/a-gdn-model-snapshots/v295/round11-t1-fixture.json").read_text()
    )


def test_p0_contract_has_two_fresh_promoted_s0_blocks():
    builder = load("build_round15_p0_snapshot.py")
    a = builder.contract(337)
    b = builder.contract(338)
    assert (a["block"], b["block"]) == ("A", "B")
    assert a["parent"] == b["parent"] == "ROUND14A_V295_PROMOTED_GDN_S0"
    assert a["generation_limit"] == b["generation_limit"] == 4
    assert a["graph"] is False


def test_p0_schedule_is_exact_warmup_w1_w2_drain():
    driver = load("round15_p0_driver.py")
    rows = driver.build_schedule(fixture(), version=331, block="A")
    assert [row["workload"] for row in rows] == ["WARMUP", "W1", "W2", "DRAIN"]
    assert [(row["input_tokens"], row["decode_tokens"]) for row in rows] == [
        (128, 16), (128, 16), (128, 128), (27, 4)
    ]
    assert len({row["request"]["rid"] for row in rows}) == 4


def test_launcher_patch_keeps_graph_off_and_promoted_gdn_on():
    builder = load("build_round15_p0_snapshot.py")
    parent = ROOT.parent / "round14a/a-gdn-model-snapshots/v295/source/tools/launch_hybrid.py"
    patched = builder.patch_launcher(parent.read_text(), builder.contract(333))
    assert "ROUND14_GDN_BACKEND_ENABLE=1" in patched
    assert "ROUND13_ATTRIBUTION_ENABLE=1" in patched
    assert "ROUND15_P0_PROFILE=1" in patched
    assert "ROUND14A_FIVE_BUCKET_ENABLE=1" in patched
    assert "ROUND14A_FIVE_BUCKET_PATH=/out/round14a-five-bucket-memory.json" in patched
    assert "ROUND15_GRAPH_ENABLE" not in patched
    assert "budget.get('stage')!='ROUND15_P0_FRESH_PROMOTED_S0_PROFILE'" in patched
    assert "wrong Round14a GDN model budget" not in patched
    compile(patched, "launch_hybrid.py", "exec")


def test_p0_builder_exposes_existing_five_bucket_overlay_installer():
    builder = load("build_round15_p0_snapshot.py")
    assert hasattr(builder, "install_five_bucket_census")
    source = (
        "def model_ready(self):\n"
        "        from sglang.srt.layers.round9.streaming_binding import streaming_model_ready\n"
        "        streaming_model_ready(self)\n"
        "        from sglang.srt.layers.round9.round14a_gdn_branch_binding import gdn_branch_model_ready\n"
        "        gdn_branch_model_ready(self)\n"
    )
    patched = builder.patch_model_runner_for_five_bucket(source)
    assert patched.count("round14a_five_bucket.capture(self)") == 1
    assert patched.index("round14a_five_bucket.capture(self)") > patched.index(
        "gdn_branch_model_ready(self)"
    )


def test_p0_preflight_rejects_nonpromoted_qwen_source(tmp_path):
    preflight = load("round15_p0_preflight.py")
    qwen = tmp_path / "qwen4_exp.py"
    qwen.write_text("wrong")
    try:
        preflight.verify_qwen(qwen)
    except RuntimeError as error:
        assert "promoted Qwen source" in str(error)
    else:
        raise AssertionError("source gate accepted the wrong Qwen implementation")


def test_p0_preflight_rejects_missing_five_bucket_producer(tmp_path):
    preflight = load("round15_p0_preflight.py")
    launcher = tmp_path / "launch_hybrid.py"
    runner = tmp_path / "model_runner.py"
    module = tmp_path / "round14a_five_bucket.py"
    launcher.write_text("ROUND15_P0_PROFILE=1\n")
    runner.write_text("streaming_model_ready(self)\n")
    try:
        preflight.verify_five_bucket_census(launcher, runner, module)
    except RuntimeError as error:
        assert "five-bucket" in str(error)
    else:
        raise AssertionError("source gate accepted a missing five-bucket producer")


def test_p0_driver_normalizes_profiler_contract():
    driver = load("round15_p0_driver.py")
    result = {"profiler": "BOUNDED_OBSERVER_ON", "status": "PASS_ROUND14A_B_PROFILE_DRIVER_RAW"}
    normalized = driver.normalize_result(result)
    assert normalized is result
    assert normalized["profiler"] == "BOUNDED_EXISTING_EVENT_OBSERVER"


def test_p0_cleanup_accepts_only_registered_training_process():
    control = load("round15_p0_control.py")
    proc = "\tProcess id:2980236 Process name:python3.10        Process memory(MB):6123\n"
    cgroups = {
        2980236: (
            "0::/docker/"
            "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef\n"
        )
    }
    result = control.classify_physical12_after(proc, cgroups.__getitem__)
    assert result["status"] == "REGISTERED_TRAINING_ONLY"
    assert result["processes"] == [{"pid": 2980236, "memory_mib": 6123}]


def test_p0_cleanup_rejects_unknown_device_process():
    control = load("round15_p0_control.py")
    proc = "Process id:42 Process name:unknown Process memory(MB):1024\n"
    try:
        control.classify_physical12_after(proc, lambda _pid: "0::/docker/" + "f" * 64 + "\n")
    except RuntimeError as error:
        assert "unknown physical12 process" in str(error)
    else:
        raise AssertionError("unknown device process was accepted")


def test_p0_cleanup_accepts_empty_device():
    control = load("round15_p0_control.py")
    result = control.classify_physical12_after("No process in device\n", lambda _pid: "")
    assert result == {"status": "EMPTY", "processes": []}


def test_p0_controller_missing_container_requires_supervisor_cleanup_proof():
    control = load("round15_p0_control.py")
    container_id = "8" * 64
    proof = control.validate_supervisor_removed_container(
        {
            "container_id": container_id,
            "owned_container_removed": True,
            "guard_exit_ack": {"action": "OWNED_EXITED_DURING_SAMPLE", "time": 123.0},
        },
        container_id,
    )
    assert proof["status"] == "ALREADY_REMOVED_BY_SUPERVISOR"
    try:
        control.validate_supervisor_removed_container(
            {"container_id": container_id, "owned_container_removed": False}, container_id
        )
    except RuntimeError as error:
        assert "supervisor removal proof" in str(error)
    else:
        raise AssertionError("missing container was accepted without supervisor proof")
