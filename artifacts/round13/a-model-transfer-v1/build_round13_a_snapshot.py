#!/usr/bin/env python3
"""Build one immutable Round13 A model arm from Round12a v177."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


ROLES = {
    196: ("uniform", "A", "uniform-a"), 197: ("frequency", "A", "frequency-a"),
    198: ("front_loading", "A", "front-a"), 199: ("random_seed42", "A", "random-a"),
    200: ("random_seed42", "B", "random-b"), 201: ("front_loading", "B", "front-b"),
    202: ("frequency", "B", "frequency-b"), 203: ("uniform", "B", "uniform-b"),
}
MUTABLE = {"round13-source-sha256.json", "execution-budget.json",
           "execution-budget.lock", "host-controller.json"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new, label):
    if source.count(old) != 1: raise ValueError(f"{label} precondition mismatch")
    return source.replace(old, new)


def patch_framework(source):
    source = replace_once(source,
        "    config = load_framework()\n    if config is None: return 64\n",
        "    config = load_framework()\n    if config is None: return 64\n"
        "    from .round13_placement_runtime import profile_from_environment, bind_masks\n"
        "    round13_profile = profile_from_environment()\n"
        "    if round13_profile is not None:\n"
        "        return bind_masks(args, wrapper, round13_profile)\n",
        "Round13 prepare_masks binding")
    old = (
        "    document = collect_q32_physical_layers(\n"
        "        scheduler.tp_worker.model_runner.model, KTEPWrapperMethod)\n")
    new = (
        "    from .round13_placement_runtime import profile_from_environment, collect_physical_layers\n"
        "    round13_profile = profile_from_environment()\n"
        "    document = (collect_physical_layers(\n"
        "        scheduler.tp_worker.model_runner.model, KTEPWrapperMethod, round13_profile)\n"
        "        if round13_profile is not None else collect_q32_physical_layers(\n"
        "        scheduler.tp_worker.model_runner.model, KTEPWrapperMethod))\n")
    source = replace_once(source, old, new, "Round13 physical recorder")
    compile(source, "framework_hooks.py", "exec")
    return source


def patch_qwen(source):
    old = (
        "        from sglang.srt.layers.round9.candidate_config import validate_kt_config, bind_model_profile\n"
        "        from sglang.srt.layers.round9.candidate_bank import bind_verified_banks\n"
        "        from sglang.srt.layers.round9.dynamic_boundary import installed_scheduler_boundary\n"
        "        candidate = validate_kt_config(cfg, json.loads(Path(candidate_path).read_text()),\n"
        "            native_bf16_tail=bool(getattr(kt_kernel_ext,\n"
        "                \"llamafile_bf16_f32_tail_supported\", lambda: False)()),\n"
        "            native_block32_tail=bool(getattr(kt_kernel_ext,\n"
        "                \"llamafile_block32_tail_supported\", lambda: False)()),\n"
        "            dynamic_boundary_installed=installed_scheduler_boundary(),\n"
        "            resident_capacity=round9_resident_capacity)\n"
        "        bank_identity = bind_verified_banks(candidate_path, args.kt_weight_path, args.model_path)\n"
        "        from sglang.srt.layers.round9.framework_hooks import validate_q32_model_gate\n"
        "        validate_q32_model_gate(round9_resident_capacity, candidate, bank_identity)\n"
        "        bind_model_profile(candidate_path, layer_id, bank_identity)\n"
        "        return\n")
    new = (
        "        from sglang.srt.layers.round9.candidate_config import (\n"
        "            Candidate, validate_kt_config, bind_model_profile)\n"
        "        from sglang.srt.layers.round9.candidate_bank import bind_verified_banks\n"
        "        from sglang.srt.layers.round9.dynamic_boundary import installed_scheduler_boundary\n"
        "        from sglang.srt.layers.round9.round13_placement_runtime import (\n"
        "            profile_from_environment, validate_layer_config, require_component_gate)\n"
        "        round13_profile = profile_from_environment()\n"
        "        candidate_document = json.loads(Path(candidate_path).read_text())\n"
        "        if round13_profile is not None:\n"
        "            candidate = Candidate.from_dict(candidate_document)\n"
        "            if (candidate.workers != 128 or candidate.storage != 'MXFP4' or\n"
        "                    candidate.placement != 'static' or tuple(candidate.numa_nodes) != (2,3,4,7) or\n"
        "                    not bool(getattr(kt_kernel_ext, 'llamafile_block32_tail_supported', lambda: False)())):\n"
        "                raise RuntimeError('Round13 placement requires frozen C128/MXFP4 static candidate')\n"
        "            validate_layer_config(cfg, layer_id, round13_profile, args)\n"
        "            require_component_gate(os.environ['ROUND13_PLACEMENT_COMPONENT_GATE'],\n"
        "                os.environ['ROUND13_PLACEMENT_COMPONENT_GATE_SHA256'])\n"
        "        else:\n"
        "            candidate = validate_kt_config(cfg, candidate_document,\n"
        "                native_bf16_tail=bool(getattr(kt_kernel_ext,\n"
        "                    'llamafile_bf16_f32_tail_supported', lambda: False)()),\n"
        "                native_block32_tail=bool(getattr(kt_kernel_ext,\n"
        "                    'llamafile_block32_tail_supported', lambda: False)()),\n"
        "                dynamic_boundary_installed=installed_scheduler_boundary(),\n"
        "                resident_capacity=round9_resident_capacity)\n"
        "        bank_identity = bind_verified_banks(candidate_path, args.kt_weight_path, args.model_path)\n"
        "        from sglang.srt.layers.round9.framework_hooks import validate_q32_model_gate\n"
        "        validate_q32_model_gate(round9_resident_capacity, candidate, bank_identity)\n"
        "        bind_model_profile(candidate_path, layer_id, bank_identity)\n"
        "        return\n")
    source = replace_once(source, old, new, "Round13 Qwen per-layer candidate gate")
    source = replace_once(source, "def _check_kt_layer(mlp):\n", "def _check_kt_layer(mlp, layer_id):\n",
                          "Round13 Qwen method gate signature")
    old_check = (
        "    from sglang.srt.layers.moe.quant_method_registry import is_wrapped_method\n"
        "    if not is_wrapped_method(mlp.experts.quant_method, \"kt_ep\"):\n"
        "        raise RuntimeError(\"Round9 sparse layer was not constructed with KTEP\")\n")
    new_check = (
        "    from sglang.srt.layers.moe.quant_method_registry import is_wrapped_method\n"
        "    from sglang.srt.layers.round9.round13_placement_runtime import (\n"
        "        profile_from_environment, require_method_identity)\n"
        "    round13_profile = profile_from_environment()\n"
        "    if round13_profile is not None:\n"
        "        require_method_identity(mlp.experts.quant_method, layer_id, round13_profile, is_wrapped_method)\n"
        "    elif not is_wrapped_method(mlp.experts.quant_method, \"kt_ep\"):\n"
        "        raise RuntimeError(\"Round9 sparse layer was not constructed with KTEP\")\n")
    source = replace_once(source, old_check, new_check, "Round13 Qwen provider identity")
    if source.count("_check_kt_layer(self.mlp)") != 2:
        raise ValueError("Round13 Qwen call count mismatch")
    source = source.replace("_check_kt_layer(self.mlp)", "_check_kt_layer(self.mlp, layer_id)")
    compile(source, "qwen4_exp.py", "exec")
    return source


def patch_launcher(source, version, strategy, block, label):
    source = replace_once(source, "if a.version != 177 or a.cost_mode != 'OFF':",
                          f"if a.version != {version} or a.cost_mode != 'OFF':", "launcher version")
    source = replace_once(source,
        "name='kt-round12a-kt-b-v177';out=root/'evidence'/'20260927-round12a-kt-b-v177'",
        f"name='kt-round13-a-{label}-v{version}';out=root/'evidence'/'20260927-round13-a-{label}-v{version}'",
        "launcher identity")
    source = replace_once(source,
        "if budget.get('stage')!='ROUND12A_ABBA':raise RuntimeError('wrong Round12a ABBA budget')",
        "if budget.get('stage')!='ROUND13_A_STATIC_PLACEMENT':raise RuntimeError('wrong Round13 A budget')",
        "launcher budget")
    anchor = "args+=['-e','ROUND9_HYBRID_PROBE=1']\n"
    env = (anchor +
        f"args+=['-e','ROUND13_PLACEMENT_PROFILE=/round9/round13-a-{label}-v{version}/PLACEMENT_PROFILES.json']\n"
        f"args+=['-e','ROUND13_PLACEMENT_PROFILE_SHA256='+digest(root/'round13-a-{label}-v{version}/PLACEMENT_PROFILES.json')]\n"
        f"args+=['-e','ROUND13_PLACEMENT_STRATEGY={strategy}']\n"
        f"args+=['-e','ROUND13_PLACEMENT_COMPONENT_GATE=/round9/round13-a-{label}-v{version}/A_COMPONENT_GATE.json']\n"
        f"args+=['-e','ROUND13_PLACEMENT_COMPONENT_GATE_SHA256='+digest(root/'round13-a-{label}-v{version}/A_COMPONENT_GATE.json')]\n")
    source = replace_once(source, anchor, env, "launcher placement environment")
    compile(source, "launch_hybrid.py", "exec")
    return source


def source_manifest(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and str(path.relative_to(root)) not in MUTABLE
        and not str(path.relative_to(root)).startswith("host-controller")}


def build(args):
    strategy, block, label = ROLES[args.version]
    if args.output.exists(): raise FileExistsError(args.output)
    shutil.copytree(args.parent, args.output, ignore=shutil.ignore_patterns("__pycache__"))
    for path in (args.output, *args.output.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    for name in ("host-controller.json","execution-budget.lock","round12-source-sha256.json",
                 "ROUND12A_MODEL_CONTRACT.json","ROUND12A_ABBA_QUEUE.json"):
        path=args.output/name
        if path.exists(): path.unlink()
    copies = {
        "round13_placement_runtime.py":"overlay-python/sglang/srt/layers/round9/round13_placement_runtime.py",
        "round13_a_model_driver.py":"round13_a_model_driver.py",
        "round13_model_controller.py":"round13_model_controller.py",
        "round13_model_supervisor.py":"round13_model_supervisor.py",
        "round13_monitor_owned.py":"source/tools/monitor_owned_server.py",
        "round11_t4_driver.py":"round11_t4_driver.py",
        "round10_t1_driver.py":"round10_t1_driver.py",
        "round10_t1_protocol.py":"round10_t1_protocol.py",
        "round10-t1-fixture.json":"round10-t1-fixture.json",
        "round11-t1-fixture.json":"round11-t1-fixture.json",
        "PLACEMENT_PROFILES.json":"PLACEMENT_PROFILES.json",
        "A_COMPONENT_GATE.json":"A_COMPONENT_GATE.json",
    }
    for source,target in copies.items(): shutil.copy2(args.inputs/source,args.output/target)
    framework=args.output/"overlay-python/sglang/srt/layers/round9/framework_hooks.py"
    framework.write_text(patch_framework(framework.read_text()))
    qwen=args.output/"overlay-python/sglang/srt/models/qwen4_exp.py"
    qwen.write_text(patch_qwen(qwen.read_text()))
    overlay=args.output/"overlay-python"
    sys.path.insert(0,str(args.output/"source/tools"))
    try:
        from install_candidate_overlay import tree_digest,tree_files
        receipt_path=overlay/"round9-candidate-overlay.json";receipt=json.loads(receipt_path.read_text())
        receipt["candidate_files"]=tree_files(overlay);receipt["candidate_tree_sha256"]=tree_digest(receipt["candidate_files"])
        receipt["round13_a"]={"status":"INSTALLED_FROZEN_GLOBAL_BUDGET_PLACEMENT",
            "strategy":strategy,"framework_sha256":sha(framework),"qwen_sha256":sha(qwen)}
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
    finally: sys.path.remove(str(args.output/"source/tools"))
    launcher=args.output/"source/tools/launch_hybrid.py"
    launcher.write_text(patch_launcher(launcher.read_text(),args.version,strategy,block,label))
    for name in ("framework-profile.json","streaming-profile.json"):
        path=args.output/name;doc=json.loads(path.read_text())
        if name.startswith("framework"):
            if doc["prefill"]["threshold_tokens"]!=129: raise ValueError("parent framework threshold")
            doc["prefill"]["threshold_tokens"]=513
        else:
            if doc["threshold_tokens"]!=129: raise ValueError("parent streaming threshold")
            doc["threshold_tokens"]=513
        path.write_text(json.dumps(doc,indent=2)+"\n")
    budget={"schema":1,"stage":"ROUND13_A_STATIC_PLACEMENT","version":args.version,
        "strategy":strategy,"block":block,"model_starts_limit":1,"model_starts_used":0,
        "starts":[],"generation_limit":24,"abort_limit":1 if block=="A" else 0,
        "automatic_retries_limit":0}
    (args.output/"execution-budget.json").write_text(json.dumps(budget,indent=2)+"\n")
    contract={"schema":1,"status":"FROZEN_NOT_RUN","stage":"ROUND13_A_STATIC_PLACEMENT",
        "version":args.version,"strategy":strategy,"block":block,
        "container":f"kt-round13-a-{label}-v{args.version}",
        "evidence":f"20260927-round13-a-{label}-v{args.version}",
        "generation_limit":24,"abort_limit":1 if block=="A" else 0,"automatic_retries":0,
        "profiler":"OFF","raw_result":"round13-a-raw.json","final_result":"round13-a-final.json",
        "driver_pass_status":"PASS_ROUND13_A_MODEL_DRIVER_RAW",
        "process_pass_status":"PASS_ROUND13_A_MODEL_PROCESS",
        "driver_command":["{DRIVER_ROOT}/round13_a_model_driver.py","--version",str(args.version),
            "--strategy",strategy,"--block",block,"--out","/out",
            "--profiles","{DRIVER_ROOT}/PLACEMENT_PROFILES.json",
            "--fixture10","{DRIVER_ROOT}/round10-t1-fixture.json",
            "--fixture11","{DRIVER_ROOT}/round11-t1-fixture.json"]}
    (args.output/"ROUND13_MODEL_CONTRACT.json").write_text(json.dumps(contract,indent=2)+"\n")
    (args.output/"round13-source-sha256.json").write_text(json.dumps(source_manifest(args.output),indent=2,sort_keys=True)+"\n")
    print("PASS_ROUND13_A_SNAPSHOT",args.version,strategy,block,len(source_manifest(args.output)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--version",type=int,choices=tuple(ROLES),required=True);build(p.parse_args())


if __name__=="__main__":main()
