import asyncio
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace as NS
import unittest


class CancelOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tools = Path(__file__).resolve().parents[1] / "tools"
        sys.path.insert(0, str(self.tools))

    def tearDown(self):
        sys.path.remove(str(self.tools))

    def fake_module(self, **functions):
        name = "sglang.srt.layers.round9.cancel_lifecycle"
        old = sys.modules.get(name)
        module = types.ModuleType(name)
        for key, value in functions.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return name, old

    def restore_module(self, name, old):
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old

    def test_tokenizer_seam_records_send_decision_without_changing_behavior(self):
        from install_framework_overlay import patch_tokenizer_manager

        source = '''
class Tokenizer:
    def __init__(self):
        self.rid_to_state = {"live": object()}
        self.sent = []
        self.enable_metrics = False
        self.send_to_scheduler = type("Sender", (), {"send_pyobj": lambda owner, req: self.sent.append(req)})()
    def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:
            return
        req = (rid, abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            pass
'''
        decisions = []
        name, old = self.fake_module(
            tokenizer_abort_requested=lambda manager, rid, abort_all:
                decisions.append((rid, abort_all, rid in manager.rid_to_state))
        )
        try:
            scope = {}
            patched = patch_tokenizer_manager(source)
            exec(compile(patched, "tokenizer_manager.py", "exec"), scope)
            manager = scope["Tokenizer"]()
            manager.abort_request("missing")
            manager.abort_request("live")
        finally:
            self.restore_module(name, old)

        self.assertEqual(decisions, [
            ("missing", False, False),
            ("live", False, True),
        ])
        self.assertEqual(manager.sent, [("live", False)])
        with self.assertRaises(ValueError):
            patch_tokenizer_manager(patched)

    def test_scheduler_seam_marks_cancel_before_upstream_to_finish(self):
        from install_framework_overlay import patch_scheduler

        source = '''
class Scheduler:
    def event_loop_normal(self):
        while False:
            if batch:
                scheduler_before(self, batch)
                result = self.run_batch(batch)
            scheduler_after(self)
            self.last_batch = batch
    def abort_request(self, recv_req: object):
        self.order.append("upstream")
        for req in []:
            req.to_finish = FINISH_ABORT()

    def _pause_engine(self):
        pass
'''
        order = []
        name, old = self.fake_module(
            scheduler_batch_submit_fence=lambda scheduler, batch: order.append("batch-fence"),
            scheduler_abort_received=lambda scheduler, req: order.append("seen"),
            scheduler_abort_finished=lambda scheduler, req: order.append("finalized"),
        )
        try:
            scope = {"scheduler_before": lambda *args: None,
                     "scheduler_after": lambda *args: None}
            patched = patch_scheduler(source)
            exec(compile(patched, "scheduler.py", "exec"), scope)
            scheduler = scope["Scheduler"]()
            scheduler.order = order
            scheduler.abort_request(object())
        finally:
            self.restore_module(name, old)

        self.assertEqual(order, ["seen", "upstream", "finalized"])
        self.assertIn("scheduler_batch_submit_fence(self, batch)", patched)
        with self.assertRaises(ValueError):
            patch_scheduler(patched)

    def test_http_seam_records_receipt_before_manager_dispatch(self):
        from install_framework_overlay import patch_http_server

        source = '''
async def abort_request(obj, request):
    """Abort a request."""
    try:
        manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return 200
    except Exception:
        return 500
'''
        order = []
        name, old = self.fake_module(
            http_abort_received=lambda rid, abort_all: order.append(("http", rid, abort_all))
        )
        try:
            scope = {"manager": NS(abort_request=lambda **kwargs: order.append(("manager", kwargs)))}
            patched = patch_http_server(source)
            exec(compile(patched, "http_server.py", "exec"), scope)
            result = asyncio.run(scope["abort_request"](NS(rid="r", abort_all=False), object()))
        finally:
            self.restore_module(name, old)

        self.assertEqual(result, 200)
        self.assertEqual(order, [
            ("http", "r", False),
            ("manager", {"rid": "r", "abort_all": False}),
        ])
        with self.assertRaises(ValueError):
            patch_http_server(patched)

    def test_candidate_launcher_exposes_default_off_cancel_trace_flag(self):
        launcher = self.tools / "launch_candidate_model.py"
        result = subprocess.run(
            [sys.executable, str(launcher), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertIn("--cancel-trace", result.stdout)
        self.assertIn("--cancel-closure", result.stdout)
        self.assertIn("--working-set-timeline", result.stdout)

    def test_candidate_launcher_rejects_timeline_without_streaming_stack(self):
        launcher = self.tools / "launch_candidate_model.py"
        result = subprocess.run(
            [sys.executable, str(launcher),
             "--version", "54", "--storage", "MXFP4", "--placement", "static",
             "--bank-receipt", "/missing-bank", "--device-gate", "/missing-gate",
             "--overlay-python", "/missing-overlay", "--working-set-timeline"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "working-set timeline requires the bounded three-plane streaming profile",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
