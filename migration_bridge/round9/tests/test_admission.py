"""Admission must fail closed before an owned NPU workload is launched."""
import copy
import importlib.util
import pathlib
import unittest

TOOL = pathlib.Path(__file__).parents[1] / 'tools' / 'admission.py'


class AdmissionTests(unittest.TestCase):
    def evaluate(self, snapshot, profile_id="a3-physical15"):
        self.assertTrue(TOOL.exists(), 'admission evaluator is missing')
        spec = importlib.util.spec_from_file_location('admission', TOOL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluate(snapshot, now=1000, profile_id=profile_id)

    def valid(self):
        return dict(host="110.123.0.3", pci_bus_id="0000:83:00.0", logical_device=0, physical_id=15, device_nodes=[15], mapping_verified=True,
                    access_verified=True, identity_status='PASS',
                    prerequisite_statuses={'runtime': 'PASS'},
                    sampled_at=995, hbm_total_mib=65536,
                    training_peak_mib=10000, owned_peak_budget_mib=4096,
                    reserve_mib=12288, cpu_physical_keys=['0:1','0:2'],
                    cpu_set=[2,4], observed_cpu_keys={'2':'0:1','4':'0:2'},
                    telemetry_ok=True)

    def test_valid_bounded_probe(self):
        self.assertEqual(self.evaluate(self.valid())['status'], 'PASS')

    def test_rejects_wrong_or_additional_device(self):
        for change in ({'physical_id':14},{'device_nodes':[14,15]},
                       {'mapping_verified':False},{'access_verified':False}):
            with self.subTest(change=change):
                x=self.valid(); x.update(change)
                self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_missing_fields_fail_closed(self):
        for key in self.valid():
            with self.subTest(key=key):
                x=self.valid(); del x[key]
                self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_bad_telemetry_and_stale_or_future_timestamp(self):
        for change in ({'telemetry_ok':False},{'sampled_at':900},
                       {'sampled_at':1001},{'training_peak_mib':None},
                       {'training_peak_mib':float('nan')},
                       {'owned_peak_budget_mib':-1}):
            with self.subTest(change=change):
                x=self.valid(); x.update(change)
                self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_overcommitted_hbm(self):
        x=self.valid(); x['owned_peak_budget_mib']=50000
        self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_smt_siblings_and_forged_physical_keys(self):
        for change in ({'cpu_physical_keys':['0:1','0:1']},
                       {'cpu_set':[2,2]}, {'cpu_set':[2,3]},
                       {'observed_cpu_keys':{}}):
            with self.subTest(change=change):
                x=self.valid(); x.update(change)
                self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_identity_and_prerequisites(self):
        for change in ({'identity_status':'NOT_RUN'},
                       {'prerequisite_statuses':{}},
                       {'prerequisite_statuses':{'unrelated':'PASS'}},
                       {'prerequisite_statuses':{'runtime':'NOT_RUN'}}):
            x=self.valid(); x.update(change)
            self.assertEqual(self.evaluate(x)['status'],'BLOCKED')

    def test_return22_explicit_profile(self):
        x=self.valid(); x.update(host='110.129.0.22', pci_bus_id='0000:42:00.0',
                                physical_id=7, device_nodes=[7])
        self.assertEqual(self.evaluate(x, 'return22-device7')['status'], 'PASS')
        self.assertEqual(self.evaluate(x)['status'], 'BLOCKED')
        for change in ({'host':'110.123.0.3'}, {'physical_id':0},
                       {'pci_bus_id':'0000:C1:00.0'}, {'device_nodes':[0,7]},
                       {'logical_device':7}, {'logical_device':False}):
            y=copy.deepcopy(x); y.update(change)
            with self.subTest(change=change):
                self.assertEqual(self.evaluate(y, 'return22-device7')['status'], 'BLOCKED')
        self.assertEqual(self.evaluate(x, 'unknown')['status'], 'BLOCKED')

    def test_does_not_modify_input(self):
        x=self.valid(); before=copy.deepcopy(x); self.evaluate(x)
        self.assertEqual(x,before)


if __name__ == '__main__':
    unittest.main()
