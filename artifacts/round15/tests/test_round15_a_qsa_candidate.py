import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((
        ROOT.parent / "round14a/a-gdn-model-snapshots/v295/round10-t1-fixture.json"
    ).read_text())


def load_candidate():
    path = ROOT / "round15_a_qsa_candidate.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_decode_uses_exact_full_set_path():
    candidate = load_candidate()
    assert candidate.select_strategy(
        query_tokens=1,
        key_groups=64,
        topk_blocks=512,
        ratio=4,
        n_heads=4,
        head_dim=128,
        dtype_name="bfloat16",
    ) == "FULL_SET_CAUSAL"


def test_long_decode_can_screen_lightning_indexer():
    candidate = load_candidate()
    assert candidate.select_strategy(
        query_tokens=1,
        key_groups=520,
        topk_blocks=512,
        ratio=4,
        n_heads=4,
        head_dim=128,
        dtype_name="bfloat16",
    ) == "LIGHTNING_INDEXER"


def test_prefill_and_unqualified_geometry_fall_back_before_mutation():
    candidate = load_candidate()
    common = dict(
        key_groups=64,
        topk_blocks=512,
        ratio=4,
        n_heads=4,
        head_dim=128,
        dtype_name="bfloat16",
    )
    assert candidate.select_strategy(query_tokens=2, **common) == "EAGER_REFERENCE"
    assert candidate.select_strategy(query_tokens=1, **{**common, "ratio": 8}) == "EAGER_REFERENCE"
    assert candidate.select_strategy(query_tokens=1, **{**common, "n_heads": 8}) == "EAGER_REFERENCE"
    assert candidate.select_strategy(query_tokens=1, **{**common, "dtype_name": "float32"}) == "EAGER_REFERENCE"


def test_full_set_indices_cover_exact_causal_prefix_and_padding():
    candidate = load_candidate()
    for position in (0, 3, 4, 31, 127, 255):
        selected = candidate.full_set_logical_indices(
            logical_position=position,
            ratio=4,
            topk_blocks=512,
        )
        valid = [value for value in selected if value >= 0]
        assert valid == list(range(position + 1))
        assert len(selected) == 2051
        assert selected[position + 1 :] == [-1] * (2051 - position - 1)


def test_full_set_indices_reject_position_outside_exact_path():
    candidate = load_candidate()
    for position in (-1, 2048):
        try:
            candidate.full_set_logical_indices(
                logical_position=position,
                ratio=4,
                topk_blocks=512,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("full-set helper accepted an unqualified position")


def test_component_contract_freezes_two_independent_blocks():
    path = ROOT / "build_round15_a1_component_snapshot.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    first = builder.contract(339)
    second = builder.contract(340)
    assert (first["block"], second["block"]) == ("A", "B")
    assert first["candidate_default"] == second["candidate_default"] == "OFF"
    assert first["candidate_family"] == second["candidate_family"] == "QSA_INDEX_SELECTION"
    assert first["full_parent_improvement_gate"] == 0.15
    assert first["same_shape_cycles"] == 20
    assert first["graph"] is False


def test_component_guard_patch_admits_only_frozen_round15_names():
    path = ROOT / "build_round15_a1_component_snapshot.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = (
        "def owned_container_name(name):\n"
        "    return re.fullmatch(\n"
        "        r'(?:old-name)', name) is not None\n"
    )
    patched = builder.patch_guard(source)
    assert "kt-round15-a1-qsa-[ab]-v(?:339|340)" in patched
    compile(patched, "monitor_owned_server.py", "exec")


def test_two_block_comparator_requires_every_parent_ratio():
    path = ROOT / "compare_round15_a1_component.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    def block(name, ratio=0.2):
        return {
            "status": "PASS_ROUND15_A1_QSA_COMPONENT_PROCESS",
            "block": name,
            "source_identity": {"revision": "frozen"},
            "selected_component_candidate": "FULL_SET_CAUSAL_SHORT_CONTEXT",
            "full_set_rows": [
                {
                    "layer": layer,
                    "context_tokens": context,
                    "candidate_over_baseline": ratio,
                    "downstream_relative_l2": 0.0,
                    "selected_set_exact": True,
                    "repeat_hashes": ["same"] * 3,
                }
                for layer in (3, 23, 47)
                for context in (32, 64, 128, 256)
            ],
            "lifecycle_verdict": {"pass": True},
            "component_gate_pass": True,
            "peak_npu_allocated": 1024,
            "lightning_disposition": "SEMANTICALLY_EXPRESSIBLE_OUTSIDE_CURRENT_REQUEST_RANGE",
        }

    passed = comparator.compare([block("A"), block("B")])
    assert passed["status"] == "PASS_ROUND15_A1_QSA_TWO_BLOCK_COMPONENT_GATE"
    assert passed["request_smoke_gate_open"] is True
    failed = comparator.compare([block("A"), block("B", ratio=0.851)])
    assert failed["status"] == "ROUND15_A1_QSA_COMPONENT_NO_PROMOTION"
    assert failed["request_smoke_gate_open"] is False


def test_model_contract_freezes_two_process_directional_screen():
    path = ROOT / "build_round15_a_model_snapshot.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    expected = {
        341: ("screen", "baseline", "S", 14, 1),
        342: ("screen", "candidate", "S", 14, 1),
    }
    for version, values in expected.items():
        contract = builder.contract(version)
        assert (
            contract["kind"],
            contract["implementation"],
            contract["block"],
            contract["generation_limit"],
            contract["abort_limit"],
        ) == values
        assert contract["profiler"] == "OFF"
        assert contract["graph"] is False
        assert contract["candidate_default"] == "OFF"
        assert contract["request_evidence_class"] == "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA"


def test_model_candidate_patch_is_decode_only_and_default_off():
    path = ROOT / "build_round15_a_model_snapshot.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = (ROOT / "p0-snapshots/v338/overlay-python/sglang/srt/layers/round9/qsa.py").read_text()
    patched = builder.patch_qsa(source)
    assert patched.count("ROUND15_QSA_FULL_SET_ENABLE") == 1
    assert 'os.environ.get("ROUND15_QSA_FULL_SET_ENABLE", "0") == "1"' in patched
    assert "index_query.shape[0] == 1" in patched
    assert "compressed_keys.shape[0] <= topk_blocks" in patched
    assert "layer_id=None" in patched
    compile(patched, "qsa.py", "exec")


def test_model_driver_combines_smoke_with_two_workload_directional_screen():
    path = ROOT / "round15_a_model_driver.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    data = fixture()
    baseline = driver.build_schedule(data, 341)
    candidate = driver.build_schedule(data, 342)
    assert len(baseline) == len(candidate) == 14
    assert sum(row["phase"] == "cancel" for row in baseline) == 1
    assert [row["workload"] for row in baseline if row.get("workload")] == [
        workload for workload in ("W1", "W2") for _ in range(3)
    ]
    for workload in ("W1", "W2"):
        rows = [row for row in baseline if row.get("workload") == workload]
        assert [row["phase"] for row in rows] == [
            "warmup", "measurement", "measurement"
        ]


def test_model_preflight_freezes_directional_evidence_class():
    source = (ROOT / "round15_a_model_preflight.py").read_text()
    assert '341: ("screen", "baseline", "S")' in source
    assert '342: ("screen", "candidate", "S")' in source
    assert "DIRECTIONAL_NOT_FOUR_PROCESS_ABBA" in source
    assert "generation_limit\") != 14" in source
    assert "abort_limit\") != 1" in source
    compile(source, "round15_a_model_preflight.py", "exec")
