import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluate_stateful_contract import canonical, integrity_and_exposure, main


def fixture(exposed_prompts=0):
    corpus = dict(corpus="QD3", token_budget=64,
        requests=[dict(rid=f"p{i}", input_ids=[i+1], category="fixture") for i in range(128)])
    corpus["sha256"] = canonical(corpus)
    prereg = dict(prompts_per_corpus=128, candidate_topk=32, exposure_minimum=500,
        coverage=.95, alpha_per_family=.05/3, prompt_tail_risk_budget=.05,
        corpus_sha256={"QD3":corpus["sha256"]})
    manifests = []
    for mode in ("defer0", "defer1"):
        rows = []
        for i, request in enumerate(corpus["requests"]):
            rid, prompt = request["rid"], request["input_ids"]
            for t in range(64):
                exposed = mode == "defer1" and i < exposed_prompts
                routes = [dict(event_id=f"{rid}:first", request_id=rid, producer_layer=7,
                    consumer_layer=8, expert_logical_id=16, topk_rank=5, token_position=0,
                    scope="current" if t == 0 else "prefix", route_score=.1,
                    normalized_route_weight=.1)] if exposed else []
                rows.append(dict(position_id=f"{rid}:{t}", request_id=rid, prompt_id=rid,
                    prompt_hash=canonical(prompt), history_hash=canonical([2]*t),
                    request_input_hash=canonical(prompt+[2]*t), token_index=t,
                    current_input_position=t, forward_id=f"123:{i*65+t}",
                    logits_sha256="a"*64, logits_shape=[102400], mode=mode,
                    baseline_history_sha256="b"*64, trace_complete=True, pending_drained=True,
                    deferred_event_count_current_row=int(exposed and t == 0),
                    deferred_event_count_prefix=int(exposed and t > 0),
                    cpu_route_count_current_row=1, cpu_route_count_prefix=t,
                    deferred_route_records=routes,
                    deferred_exposure=("CURRENT_EXPOSURE" if t == 0 else "PREFIX_ONLY_EXPOSURE") if exposed else "NO_EXPOSURE",
                    forced_token=2, actual_input_ids=prompt if t == 0 else [2], actual_positions=[t]))
        manifests.append(dict(mode=mode, corpus_sha256=canonical(corpus),
            baseline_history_sha256="b"*64, row_count=len(rows), rows=rows))
    return corpus, prereg, manifests


class StatefulGateTests(unittest.TestCase):
    def test_causal_exposure_count(self):
        corpus, prereg, manifests = fixture(8)
        _, count = integrity_and_exposure(corpus, prereg, *manifests)
        self.assertEqual(count, 512)

    def test_missing_position_rejected(self):
        corpus, prereg, manifests = fixture()
        manifests[1]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "COVERAGE"):
            integrity_and_exposure(corpus, prereg, *manifests)

    def test_wrong_actual_input_rejected(self):
        corpus, prereg, manifests = fixture()
        manifests[1]["rows"][1]["actual_input_ids"] = [42]
        with self.assertRaisesRegex(ValueError, "CAUSAL_TOKEN_HISTORY"):
            integrity_and_exposure(corpus, prereg, *manifests)

    def test_prefix_exposure_reset_rejected_independently(self):
        corpus, prereg, manifests = fixture(8)
        row = manifests[1]["rows"][1]
        row.update(deferred_route_records=[], deferred_event_count_prefix=0, deferred_exposure="NO_EXPOSURE")
        with self.assertRaisesRegex(ValueError, "CAUSAL_EXPOSURE_RESET"):
            integrity_and_exposure(corpus, prereg, *manifests)

    def test_wrong_previous_cpu_count_rejected(self):
        corpus, prereg, manifests = fixture()
        manifests[1]["rows"][1]["cpu_route_count_prefix"] = 0
        with self.assertRaisesRegex(ValueError, "CAUSAL_EXPOSURE_RESET"):
            integrity_and_exposure(corpus, prereg, *manifests)

    def test_insufficient_exposure_precedes_torch_import(self):
        corpus, prereg, manifests = fixture()
        import builtins
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("torch imported before exposure gate")
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, obj in (("corpus", corpus), ("prereg", prereg), ("base", manifests[0]), ("defer", manifests[1])):
                obj = {k:v for k,v in obj.items() if k != "sha256"}
                obj["sha256"] = canonical(obj)
                (root / f"{name}.json").write_text(json.dumps(obj))
            argv = ["evaluate", "derive", "--baseline", str(root/"base.json"),
                "--deferred", str(root/"defer.json"), "--corpus", str(root/"corpus.json"),
                "--preregistration", str(root/"prereg.json"), "--output", str(root/"out")]
            with patch.object(sys, "argv", argv), patch("builtins.__import__", guarded):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertEqual(exc.exception.code, 1)
            report = json.loads((root/"out/exposure_gate.json").read_text())
            self.assertFalse(report["numerical_metrics_read"])
            self.assertFalse((root/"out/candidate.json").exists())


if __name__ == "__main__":
    unittest.main()
