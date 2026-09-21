import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from migration_bridge.round9.runtime.ple_gate import require_ple_device_gate, SOURCE_FILES


class PLEGateTests(unittest.TestCase):
    def test_exact_component_receipt_and_source_identity_are_required(self):
        identity = {'model': 'original'}
        sources = {name: 'a' * 64 for name in SOURCE_FILES}
        gate = dict(schema=1, status='PASS_PLE_ASYNC_ORIGINAL_ROWS',
                    scope='COMPONENT_ONLY_NOT_MODEL_OR_OVERLAP_PERFORMANCE',
                    physical_device=15, source_identity=identity,
                    source_sha256=sources, torch_version='test', torch_npu_version='test',
                    slots=2, max_rows=512, row_width=160, dtype='torch.bfloat16',
                    pinned_bytes=327680, device_buffer_bytes=327680,
                    peak_extra_bytes=2097152,
                    checks={key:True for key in ('original_rows_exact','duplicates_exact',
                        'cross_shard_exact','consumer_stream_order','cancel_late_completion',
                        'slot_reuse','request_switch','zero_credits_after_close')})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gate.json'
            def verify(value):
                raw = json.dumps(value).encode(); path.write_bytes(raw)
                return require_ple_device_gate(path,hashlib.sha256(raw).hexdigest(),
                    source_identity=identity,source_sha256=sources,
                    torch_version='test',torch_npu_version='test')
            self.assertEqual(verify(gate),gate)
            for name, value in [('status','PENDING'),('slots',3),('max_rows',513),
                                ('peak_extra_bytes',0),('torch_version','other'),
                                ('source_identity',{}),('source_sha256',{})]:
                changed=copy.deepcopy(gate);changed[name]=value
                with self.subTest(name=name), self.assertRaises(RuntimeError):verify(changed)
            changed=copy.deepcopy(gate);changed['checks']['cancel_late_completion']=False
            with self.assertRaises(RuntimeError):verify(changed)


if __name__ == '__main__':unittest.main()
