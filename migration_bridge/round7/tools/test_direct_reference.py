import unittest
from server_command import direct_reference_args


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.args=['-m','sglang.launch_server','--context-length','8192',
            '--max-total-tokens','8192','--chunked-prefill-size','4096',
            '--max-prefill-tokens','4096','--disable-radix-cache',
            '--kt-weight-path','frozen.gguf','--kt-gpu-experts-ratio','1.0']
        self.cfg=dict(reference_r64=True,dynamic=False,deferred=0,observer=False,recorder=False)
    def test_static_lane_unchanged(self):
        self.assertIs(direct_reference_args(self.args,{}),self.args)
    def test_reference_bypasses_kt(self):
        result=direct_reference_args(self.args,self.cfg)
        self.assertFalse(any(x.startswith('--kt-') for x in result))
        self.assertIn('--disable-radix-cache',result)
    def test_reference_limits_match_historical(self):
        result=direct_reference_args(self.args,self.cfg)
        for flag,value in (('--context-length','2048'),('--max-total-tokens','4096'),
                           ('--chunked-prefill-size','2048'),('--max-prefill-tokens','2048')):
            self.assertEqual(result[result.index(flag)+1],value)
        self.assertEqual(self.args[3],'8192')
    def test_reference_rejects_features(self):
        for flag in ('dynamic','observer','recorder','deferred'):
            with self.assertRaisesRegex(ValueError,'FORBIDS_HYBRID'):
                direct_reference_args(self.args,{**self.cfg,flag:True})
